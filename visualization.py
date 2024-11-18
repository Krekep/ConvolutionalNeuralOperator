from CNO2d_time_dependent_and_foundation_model.test_and_fine_tune_utils.fine_tune_utils import (
    load_model,
)

which_example = "rich_mesh"
model = load_model(
    folder=f"TrainedModels/Time_CNO_{which_example}_1",
    which_example=which_example,
    in_dim=3,
    out_dim=2,
    label="268",
)
