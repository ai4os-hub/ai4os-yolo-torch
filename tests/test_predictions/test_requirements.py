"""Testing module for api predictions. This is a test file designed to use
pytest and prepared for some basic assertions and to add your own tests.

You can add new tests using the following structure:
```py
def test_{description for the test}(metadata):
    # Add your assertions inside the test function
    assert {statement_1 that returns true or false}
    assert {statement_2 that returns true or false}
```
The conftest.py module in the same directory includes the fixture to return
to your tests inside the argument variable `metadata` the value generated by
your function defined at `api.get_metadata`.

If your file grows in complexity, you can split it into multiple files in
the same folder. However, remember to add the prefix `test_` to the file.
"""

# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument


import json
import io


def test_prediction(test_predict):
    """Test the predict function."""
    # Access the test_predict fixture defined in conftest.py
    result, accept , task_type = test_predict
    print(result[0])

    # Assert the expected result based on the 'accept' argument

    if accept == "image/png":
        assert isinstance(result, io.BytesIO)
    else:
        if task_type in ["seg", "det"]:
            missing_keys = [
                key
                for key in ["name", "class", "box"]
                if key not in result[0][0]
            ]
            assert (
                not missing_keys
            ), f"Expected keys {missing_keys} missing in result"
            result = result[0]
        else:
            missing_keys = [
                key
                for key in ["file_name", "top5_prediction"]
                if key not in result[0].keys()
            ]
            assert (
                not missing_keys
            ), f"Expected keys {missing_keys} missing in result"
            result = json.dumps(result)


# Example to test predictions probabilities range 0.0 and 1.1
# def test_predictions_range(predictions):
#     """Tests that predictions are between 0 and 1."""
#     for prediction in predictions[0:10]:
#         assert all(0.0 <= x <= 1.1 for x in prediction)
