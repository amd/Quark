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
    def _optimize_kernel(self, module_instance: torch.nn.Module, all_inp_data_quant: List[torch.Tensor],
                         all_inp_data_float: List[torch.Tensor], all_out_data_float: List[torch.Tensor],
                         params: TrainParameters) -> Any:
        """
        Optimizes the weight (for adaquant) or its rounding mode (for adaround)
        The data is all numpy array format, which is convenient for randomly sampling to form mini-batch
        :param module_instance: Quantized wrapper module, may be re-wrapped with DataParallel
        :param all_inp_data_quant: Quantized wrapper module's input tensors from all dataset
        :param all_inp_data_float: Original float module's input tensors from all dataset
        :param all_out_data_float: Original float module's output tensors from all dataset
        :param params: Optimization parameters
        """
        quant_module = module_instance.module if isinstance(module_instance, torch.nn.DataParallel) else module_instance

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
            # Generate random indices for the batch
            indices = torch.randperm(len(all_inp_data_quant))[:params.batch_size].tolist()

            # Get a batch of input and output data
            inp_data_quant = torch.cat([all_inp_data_quant[i] for i in indices], dim=0)
            inp_data_float = torch.cat([all_inp_data_float[i] for i in indices], dim=0)
            out_data_float = torch.cat([all_out_data_float[i] for i in indices], dim=0)

            # Droped quantized input data with a ratio
            # If drop_ratio = 1, the input data is all from quantized model
            # If drop_ratio = 0, the input data is all from float model
            # Otherwise, the input data comes from a mixture of the two
            inp_data_mixed = torch.where(
                torch.rand_like(inp_data_quant) < params.drop_ratio, inp_data_quant, inp_data_float)

            # Clear gradients before optimization step
            optimizer.zero_grad()

            out_data_quant = self._module_forward(module_instance, inp_data_mixed)

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

        if self._cuda_is_available(params):
            # Clear cuda cache
            torch.cuda.empty_cache()

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
            out_data_temp = self._module_forward(quant_module, inp_data)

        recons_err = F.mse_loss(out_data_temp, out_data)

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
    def _cuda_is_available(self, params: TrainParameters) -> bool:
        """
        Check if cuda is available, it's applicable to ROCm and CUDA GPU both
        """
        return params.device.startswith("cuda") and torch.cuda.is_available()

    @classmethod
    def _assign_data_device(self, io_data: numpy.ndarray[Any, Any], params: TrainParameters) -> List[torch.Tensor]:
        """
        Pack a batched numpy ndarray to a list of numpy ndarrays and then convert it to a list of torch tensors
        """
        arrays = [numpy.expand_dims(io_data[i], axis=0) for i in range(io_data.shape[0])]

        if self._cuda_is_available(params):
            return [torch.from_numpy(array).to("cuda") for array in arrays]

        return [torch.from_numpy(array) for array in arrays]

    @classmethod
    def _assign_module_device(self, quant_module: torch.nn.Module, params: TrainParameters) -> Any:
        """
        Assign the module to target device
        :param quant_module: Quantized wrapper module
        :param params: Parameters for the training
        :return module_instance: DataParallel instance or original module
        """
        module_instance: Union[torch.nn.Module, torch.nn.DataParallel]  # type: ignore

        if self._cuda_is_available(params):
            # A simple data parallelism using torch.nn.DataParallel
            if len(params.device) > 5:  # Should be like "cuda:0" with device ids
                decive_ids = [int(i) for i in params.device[5:].split(',')]
                torch.cuda.set_device('cuda:{}'.format(decive_ids[0]))  # Setting the default GPU device
                module_instance = torch.nn.DataParallel(
                    quant_module.cuda(),  # Copy weights to device
                    device_ids=decive_ids,  # Each device will have a model copy
                    output_device=decive_ids[0])  # Device for the all-reduce
            else:
                module_instance = quant_module.to(torch.device("cuda"))
        else:
            module_instance = quant_module

        return module_instance

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

        # Replace quantizer if used adaround algorithm
        if params.algorithm == "adaround":
            self._replace_quantizer(quant_module)

        logger.info("Module (%s)->(%s) will be optimized by %s on %s", quant_module._input_name,
                    quant_module._output_name, params.algorithm, params.device)

        # Arrange arrays of different batch sizes into a single numpy array
        all_inp_data_quant = numpy.concatenate(inp_data_quant, dim=0) if isinstance(  # type: ignore
            inp_data_quant, List) else inp_data_quant
        all_inp_data_float = numpy.concatenate(inp_data_float, dim=0) if isinstance(  # type: ignore
            inp_data_float, List) else inp_data_float
        all_out_data_float = numpy.concatenate(out_data_float, dim=0) if isinstance(  # type: ignore
            out_data_float, List) else out_data_float

        # Check the metrics and adjust learning rate
        recons_err = self._recons_metrics(quant_module, torch.from_numpy(all_inp_data_quant),
                                          torch.from_numpy(all_out_data_float), params.algorithm, params)

        # Adjust batch size and learning rate for this layer
        if params.batch_size < 1 or params.batch_size > all_inp_data_quant.shape[0]:
            logger.warning(f"The batch size {params.batch_size} is invalid, set it to 1")
            params.batch_size = 1

        if isinstance(params.lr_adjust,
                      (tuple, list)) and len(params.lr_adjust) == 2 and recons_err > params.lr_adjust[0]:
            logger.info("Adjust lr from %f to %f because recons error %f is greater than %f", params.lr,
                        params.lr_adjust[1], recons_err, params.lr_adjust[0])
            params.lr = params.lr_adjust[1]  # large error should apply large lr

        # Optimize the module
        tensors_inp_data_quant = self._assign_data_device(all_inp_data_quant, params)
        tensors_inp_data_float = self._assign_data_device(all_inp_data_float, params)
        tensors_out_data_float = self._assign_data_device(all_out_data_float, params)

        module_instance = self._assign_module_device(quant_module, params)

        self._optimize_kernel(module_instance, tensors_inp_data_quant, tensors_inp_data_float, tensors_out_data_float,
                              params)

        quant_module = module_instance.module.to('cpu') if isinstance(
            module_instance, torch.nn.DataParallel) else module_instance.to('cpu')

        # Show the metric after optimization for comparision
        recons_err_optimized = self._recons_metrics(quant_module, torch.from_numpy(all_inp_data_quant),
                                                    torch.from_numpy(all_out_data_float), params.algorithm, params)

        # Set the flag for the module's wrapper to drop the optimized weight and bias
        recons_err_diff = recons_err_optimized - recons_err
        if params.selective_update and recons_err_diff > 0:
            logger.info("Will drop the optimized weight (and bias) because there is no gain")
            quant_module._module.opt_gained = False

        logger.info("Module (%s)->(%s) recons metrics was optimized from %f to %f (diff=%f)", quant_module._input_name,
                    quant_module._output_name, recons_err, recons_err_optimized, recons_err_diff)
