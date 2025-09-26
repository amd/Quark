import pytest

from quark.torch.utils.exceptions import AppError, LossError


def test_app_error_message_and_str():
    """
    Test that AppError stores and returns the correct message,
    and that it inherits from Exception.
    """
    err = AppError("Something went wrong")
    # Verify that __str__ returns the original message
    assert str(err) == "Something went wrong"
    # Verify inheritance
    assert isinstance(err, Exception)


def test_loss_error_inherits_and_formats_message():
    """
    Test that LossError correctly formats the message with the prefix
    and inherits from AppError and Exception.
    """
    err = LossError("Invalid loss value")
    # Verify that the formatted message contains the prefix
    assert "Loss error:" in str(err)
    assert "Invalid loss value" in str(err)
    # Verify inheritance
    assert isinstance(err, LossError)
    assert isinstance(err, AppError)
    assert isinstance(err, Exception)


def test_raise_loss_error():
    """
    Test that LossError can be raised and caught with pytest.raises,
    and that the message is preserved.
    """
    with pytest.raises(LossError) as excinfo:
        raise LossError("NaN loss detected")
    # Verify that the caught exception contains the expected message
    assert "NaN loss detected" in str(excinfo.value)
