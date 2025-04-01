#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Union, List, Any

import numpy
import torch

from .train_model_loss import TrainLoss
from .train_model_param import TrainParameters

from quark.onnx.finetuning.create_torch.base_qdq_quantizers import INTQuantizer, AdaroundINTQuantizer

from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)


class ModelOptimizer:
    """
    Optimizes weight or its rounding mode for the quantized wrapper module
    """

    @classmethod
    def _module_forward(self, quant_module: torch.nn.Module, inp_data: torch.Tensor) -> Any:
        """
        Compute output of quantized wrapper module
        :param quant_module: Quantized wrapper module
        :param inp_data: The input data to be used for computing the output
        :return: output tensor after the module's forward
        """

        return quant_module.forward(inp_data)

    @classmethod
    @log_errors
    def _optimize_kernel(self, quant_module: torch.nn.Module, all_inp_data_quant: torch.Tensor,
                         all_inp_data_float: torch.Tensor, all_out_data_float: torch.Tensor,
                         params: TrainParameters) -> Any:
        """
        Optimizes the weight (for adaquant) or its rounding mode (for adaround)
        :param quant_module: Quantized wrapper module
        :param all_inp_data_quant: Quantized wrapper module's input tensors from all dataset
        :param all_inp_data_float: Original float module's input tensors from all dataset
        :param all_out_data_float: Original float module's output tensors from all dataset
        :param params: Optimization parameters
        """
        # Set up Adam optimizer with parameters
        if params.algorithm == "adaround":
            # Before optimization, set the optimized layer's rounding mode to "Soft rounding",
            # which maps alpha parameter between zero and one
            quantizer = quant_module._module.weight_quantizer
            quantizer.use_soft_rounding = True

            optimizer = torch.optim.Adam([quantizer.alpha], lr=params.lr)  # type: ignore
        elif params.algorithm == "adaquant":
            optimize_vars = [quant_module._module.weight]

            # If required updating bias, add the bias to optimizing variables
            if params.update_bias and quant_module._module.bias_quantizer is not None:
                optimize_vars.append(quant_module._module.bias)

            optimizer = torch.optim.Adam(optimize_vars, lr=params.lr)  # type: ignore
        else:
            raise NotImplementedError("Unsupported algorithm {}".format(params.algorithm))

        # Optimize the parameters
        mean_loss = 0.0
        best_loss = -1.0

        for iteration in range(params.num_iterations):
            # Generate random indices of batch size
            indices = torch.randperm(all_inp_data_quant.size(0))[:params.batch_size]

            # Get input and output activation data of batch size
            inp_data_quant = all_inp_data_quant[indices]
            inp_data_float = all_inp_data_float[indices]
            out_data_float = all_out_data_float[indices]

            # Droped quantized input data with a ratio
            inp_data = torch.where(torch.rand_like(inp_data_quant) < params.drop_ratio, inp_data_quant, inp_data_float)

            # Clear gradients before optimization step
            optimizer.zero_grad()
            if params.device.startswith("cuda"):
                dp_decive_ids = [int(i) for i in params.device[5:].split(',')]
                quant_module.to(torch.device('cuda:' + str(dp_decive_ids[0])))
                quant_module = torch.nn.DataParallel(quant_module, device_ids=dp_decive_ids)
                # Get the module's output using trained parameters
                out_data_quant = self._module_forward(quant_module, inp_data_quant)
                quant_module = quant_module.module
            elif params.device != "cpu":
                quant_module.to(params.device)
                out_data_quant = self._module_forward(quant_module, inp_data_quant)
            else:
                out_data_quant = self._module_forward(quant_module, inp_data_quant)
            # Calculate total loss
            recons_loss = TrainLoss.calc_recon_loss(out_data_quant, out_data_float)
            if params.algorithm == "adaround":
                round_loss = TrainLoss.calc_round_loss(quantizer.alpha, params, iteration)
                total_loss = recons_loss + round_loss
            else:
                total_loss = recons_loss

            # Check if early stop or not
            # It reused the params.num_batches and params.warm_start
            num_batches = params.num_batches if params.num_batches > 1 else params.num_iterations / 10
            if params.early_stop and iteration >= params.num_iterations * params.warm_start:
                if iteration % num_batches == num_batches - 1:
                    # Average loss of a certain number of batches
                    mean_loss = mean_loss / num_batches

                    if best_loss < 0:
                        best_loss = mean_loss
                    elif mean_loss < best_loss:
                        best_loss = mean_loss
                    else:
                        logger.info(
                            "%s Iterations=%d, mean loss %5f (in %d batches) is not better than best loss %5f, early stop",
                            params.algorithm, iteration, mean_loss, num_batches, best_loss)
                        break

                    # Clear for the next accumulation
                    mean_loss = 0.0
                else:
                    # Accumulate loss in a certain number of batches
                    if params.algorithm == "adaround":
                        mean_loss += float(round_loss)
                    else:
                        mean_loss += float(recons_loss)

            # Back propagate and Update the parameter
            total_loss.backward()
            optimizer.step()

            # Show log
            if iteration % params.log_period == 0 or iteration == params.num_iterations - 1:
                learning_rate = optimizer.param_groups[0]['lr']
                if params.algorithm == "adaround":
                    logger.info("%s iterations=%d, lr=%f, loss=%5f (Recons loss=%5f, Rounding loss=%5f)",
                                params.algorithm, iteration, learning_rate, float(total_loss), float(recons_loss),
                                float(round_loss))
                else:
                    logger.info("%s iterations=%d, lr=%f, loss=%5f", params.algorithm, iteration, learning_rate,
                                float(total_loss))

        if params.algorithm == "adaround":
            # After optimization, set the optimized layer's rounding mode to "Hard rounding",
            # which maps to exact zero and one
            quantizer.use_soft_rounding = False
        if 'cuda' in params.device and torch.device("cuda:0").type:
            # Clear cuda cache
            torch.cuda.empty_cache()

    @classmethod
    def _concat_tensors(self, io_data: Union[numpy.ndarray[Any, Any], List[numpy.ndarray[Any, Any]]]) -> torch.Tensor:
        """
        Pack the numpy ndarray or list to be a single torch tensor by concating them in batch dim
        """
        if isinstance(io_data, List):
            all_data = []
            for data in io_data:
                all_data.append(torch.tensor(data).cpu())
            return torch.cat(all_data, dim=0)
        else:
            return torch.tensor(io_data).cpu()

    @classmethod
    def _set_soft_rounding(self, quant_module: torch.nn.Module, soft_enabled: bool) -> None:
        """
        Set the quantizer to use soft rounding or hard rounding
        """
        quantizer = quant_module._module.weight_quantizer
        quantizer.use_soft_rounding = soft_enabled

    @classmethod
    def _calc_recons_metrics(self, quant_module: torch.nn.Module, inp_data: torch.Tensor, out_data: torch.Tensor,
                             params: TrainParameters) -> float:
        """
        Compute mean square error of output activations
        """
        import torch.nn.functional as F

        with torch.no_grad():
            if params.device.startswith("cuda"):
                dp_decive_ids = [int(i) for i in params.device[5:].split(',')]
                quant_module.to(torch.device('cuda:' + str(dp_decive_ids[0])))
                quant_module = torch.nn.DataParallel(quant_module, device_ids=dp_decive_ids)
                # Get the module's output using trained parameters
                out_data_temp = self._module_forward(quant_module, inp_data)
                quant_module = quant_module.module
            elif params.device != "cpu":
                quant_module.to(params.device)
                out_data_temp = self._module_forward(quant_module, inp_data)
            else:
                out_data_temp = self._module_forward(quant_module, inp_data)

        if isinstance(out_data_temp, List):
            recons_err = F.mse_loss(out_data_temp[0], out_data)
        else:
            recons_err = F.mse_loss(out_data_temp, out_data)

        del out_data_temp

        return float(recons_err)

    @classmethod
    def _recons_metrics(self, quant_module: torch.nn.Module, inp_data: torch.Tensor, out_data: torch.Tensor,
                        algorithm: str, params: TrainParameters) -> float:
        """
        Compute mean square error of output activations
        :param quant_module: Quantized wrapper module
        :param inp_data: Input data to quantized wrapper module
        :param out_data: Output data from the original float module
        :param algorithm: Using hard rounding and soft rounding if algorithm is adaround
        :return recons_err: reconstruction error
        """
        if algorithm == 'adaround':
            self._set_soft_rounding(quant_module, False)
            recons_err_hard = self._calc_recons_metrics(quant_module, inp_data, out_data, params)

            self._set_soft_rounding(quant_module, True)
            recons_err_soft = self._calc_recons_metrics(quant_module, inp_data, out_data, params)

            logger.debug("The recons error metrics using hard rounding is %f and soft rounding is %f", recons_err_hard,
                         recons_err_soft)

            return recons_err_hard  # the error of hard rounding as main metric
        else:
            recons_err = self._calc_recons_metrics(quant_module, inp_data, out_data, params)
            logger.debug("The recons error metrics is %f", recons_err)

            return recons_err

    @classmethod
    def _replace_quantizer(self, quant_module: torch.nn.Module) -> None:
        """
        Replace weight quantizer with a adaround one
        :param quant_module: Quantized wrapper module
        """
        default_quantizer = quant_module._module.weight_quantizer

        if not isinstance(default_quantizer, INTQuantizer):
            raise NotImplementedError("Can't apply adaround for non-integer quantization")
        else:
            # Create a new quantizer that has "alpha" parameter
            quantizer = AdaroundINTQuantizer(default_quantizer.scale, default_quantizer.zero_point,
                                             default_quantizer.min_q, default_quantizer.max_q,
                                             default_quantizer.ch_axis, default_quantizer.q_folded)
            # Initialize "alpha" by the weight tensor
            quantizer.initialize_alpha(quant_module._module.weight.data)

            # Replace the default quantizer with the new quantizer
            setattr(quant_module._module, "weight_quantizer", quantizer)

    @classmethod
    def run(self, quant_module: torch.nn.Module, inp_data_quant: Union[numpy.ndarray[Any, Any],
                                                                       List[numpy.ndarray[Any, Any]]],
            inp_data_float: Union[numpy.ndarray[Any, Any], List[numpy.ndarray[Any, Any]]],
            out_data_float: Union[numpy.ndarray[Any, Any], List[numpy.ndarray[Any,
                                                                              Any]]], params: TrainParameters) -> None:
        """
        Run the optimization for the target module
        :param quant_module: Quantized wrapper module which consists of a compute module and a optional act module
        :param inp_data_quant: Quantized wrapper module's input data from all dataset, single array or array list
        :param inp_data_float: Original float module's input data from all dataset, single array or array list
        :param out_data_float: Original float module's output data from all dataset, single array or array list
        :param params: Optimization parameters
        """

        # Convert input and output data to torch tensor format
        all_inp_data_quant: torch.Tensor = self._concat_tensors(inp_data_quant)
        all_inp_data_float: torch.Tensor = self._concat_tensors(inp_data_float)
        all_out_data_float: torch.Tensor = self._concat_tensors(out_data_float)

        # Replace quantizer if used adaround algorithm
        if params.algorithm == "adaround":
            self._replace_quantizer(quant_module)

        # Set device for modules and tensors
        module_instance = quant_module
        if params.device.startswith("cuda"):
            dp_decive_ids = [int(i) for i in params.device[5:].split(',')]
            all_inp_data_quant = all_inp_data_quant.to(torch.device('cuda:' + str(dp_decive_ids[0])))
            all_inp_data_float = all_inp_data_float.to(torch.device('cuda:' + str(dp_decive_ids[0])))
            all_out_data_float = all_out_data_float.to(torch.device('cuda:' + str(dp_decive_ids[0])))
        elif params.device != "cpu":
            all_inp_data_quant = all_inp_data_quant.to(torch.device(params.device))
            all_inp_data_float = all_inp_data_float.to(torch.device(params.device))
            all_out_data_float = all_out_data_float.to(torch.device(params.device))

        logger.info("Module (%s)->(%s) will be optimized by %s on %s", module_instance._input_name,
                    module_instance._output_name, params.algorithm, params.device)

        # Check the metrics and adjust learning rate
        recons_err = self._recons_metrics(module_instance, all_inp_data_quant, all_out_data_float, params.algorithm,
                                          params)

        # Adjust learning rate for the layer that have a large recons error
        if isinstance(params.lr_adjust,
                      (tuple, list)) and len(params.lr_adjust) == 2 and recons_err > params.lr_adjust[0]:
            logger.info("Adjust lr from %f to %f because recons error %f is greater than %f", params.lr,
                        params.lr_adjust[1], recons_err, params.lr_adjust[0])
            params.lr = params.lr_adjust[1]  # large error should apply large lr

        # Optimize the module
        self._optimize_kernel(module_instance, all_inp_data_quant, all_inp_data_float, all_out_data_float, params)

        # Show the metric after optimization for comparision
        recons_err_optimized = self._recons_metrics(module_instance, all_inp_data_quant, all_out_data_float,
                                                    params.algorithm, params)

        # Set the flag for the module's wrapper to drop the optimized weight and bias
        recons_err_diff = recons_err_optimized - recons_err
        if params.selective_update and recons_err_diff > 0:
            logger.info("Will drop the optimized weight (and bias) because there is no gain")
            module_instance._module.opt_gained = False

        logger.info("Module (%s)->(%s) recons metrics was optimized from %f to %f (diff=%f)",
                    module_instance._input_name, module_instance._output_name, recons_err, recons_err_optimized,
                    recons_err_diff)

        # Release memory
        del all_inp_data_quant
        del all_inp_data_float
        del all_out_data_float
