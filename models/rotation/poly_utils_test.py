from poly_utils import *

def test_convert_aligned_quadruplet_to_rotated_bbox():
    e1 = [10, 10, 20, 10, 20, 20, 10, 20]
    r1 = convert_aligned_quadruplet_to_rotated_bbox(e1)
    print(r1)

def test_draw_polygon():
    e1 = [100, 110, 230, 100, 200, 170, 100, 200]

    # drawing
    im = Image.new('RGB', (400, 400), 'white')
    draw = ImageDraw.Draw(im)
    draw.polygon(e1, outline='black')
    im.show()

def test_einsum():
    bbox = np.array([
        [10, 10, 20, 10, 20, 20, 10, 20],
        [10, 10, 20, 10, 20, 20, 10, 20]
    ], dtype=np.float32).reshape(2, 4, 2)
    R = get_2d_rotation_matrix_from_radian_batch([0.1, -0.1])
    rotated = np.einsum('ijk,imk->ijm', bbox, R)

    vanilla = np.zeros_like(bbox)  # (2, 4, 2)
    for i in range(len(bbox)):
        bbox_i = bbox[i].T  # (2, 4)
        R_i = R[i, :, :]  # (2, 2)
        rot_i = R_i @ bbox_i
        vanilla[i, :, :] = rot_i.T

    assert(np.allclose(rotated, vanilla))
    print("einsum passed")

def test_convert_quadruplet_to_rotated_bbox():
    e1 = [100, 110, 230, 100, 200, 170, 130, 200]
    # e1 = [50, 100, 100, 50, 150, 100, 100, 150]
    out = align_quadruplet_corner_to_rectangle(e1)
    out = convert_aligned_quadruplet_to_rotated_bbox([out])
    out = convert_rotated_bbox_to_quadruplet(out)
    r1 = out[0].reshape(-1).tolist()

    # drawing
    im = Image.new('RGB', (600, 600), 'white')
    draw = ImageDraw.Draw(im)
    draw.polygon(e1, outline='black')
    draw.polygon(r1, outline='red')
    im.show()


if __name__ == "__main__":
    # test_convert_aligned_quadruplet_to_rotated_bbox()
    # test_draw_polygon()
    # test_einsum()
    test_convert_quadruplet_to_rotated_bbox()
