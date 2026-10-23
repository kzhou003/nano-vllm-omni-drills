import torch

from drill import load

m = load("p18_mrope_positions")


def test_text_only_is_plain_rope():
    pos, delta = m.mrope_positions([("text", 5)])
    assert pos.tolist() == [[0, 1, 2, 3, 4]] * 3
    assert delta == 0
    assert m.decode_position(5, delta).tolist() == [5, 5, 5]


def test_text_image_text():
    pos, delta = m.mrope_positions([("text", 2), ("image", (1, 4, 4)), ("text", 3)])
    assert pos.tolist() == [
        [0, 1, 2, 2, 2, 2, 4, 5, 6],
        [0, 1, 2, 2, 3, 3, 4, 5, 6],
        [0, 1, 2, 3, 2, 3, 4, 5, 6],
    ]
    assert delta == 7 - 9
    assert m.decode_position(9, delta).tolist() == [7, 7, 7]


def test_multi_frame_rectangular_grid():
    pos, delta = m.mrope_positions([("image", (2, 2, 6)), ("text", 1)], spatial_merge=2)
    # llm grid t=2, h=1, w=3 -> 6 tokens
    assert pos[:, :6].tolist() == [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 1, 2],
    ]
    assert pos[:, 6].tolist() == [3, 3, 3]
    assert delta == 4 - 7


def test_shapes_and_dtype():
    pos, _ = m.mrope_positions([("text", 3), ("image", (1, 8, 8)), ("text", 2), ("image", (1, 2, 2))])
    assert pos.shape == (3, 3 + 16 + 2 + 1)
    assert pos.dtype == torch.long
