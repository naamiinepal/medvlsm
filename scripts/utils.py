import random
from typing import Sized


def assert_split_ratio(val_ratio: float, test_ratio: float):
    assert val_ratio >= 0, "Validation set percent should be greater than 0."
    assert test_ratio >= 0, "Test set percent should be greater than 0."
    assert (
        val_ratio + test_ratio <= 1
    ), "The sum of percent of validation set and test set should be less or equal to 1"


def get_train_val_test(total_data: Sized, val_ratio: float, test_ratio: float):
    val_size = round(val_ratio * len(total_data))
    val_data = random.sample(total_data, val_size)

    rem_data = tuple(x for x in total_data if x not in set(val_data))

    test_size = round(test_ratio * len(total_data))
    test_data = random.sample(rem_data, test_size)

    train_data = [x for x in rem_data if x not in set(test_data)]

    assert len(val_data) + len(test_data) + len(train_data) == len(
        total_data
    ), "Data split is invalid."

    return train_data, val_data, test_data
