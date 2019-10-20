import Augmentor


def performAugmentation():
    augImageDirectory = "jpgImages/aug/test"
    p = Augmentor.Pipeline(augImageDirectory)
    p.skew(probability=0.5)
    p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
    p.random_distortion(probability=0.5, grid_width=8, grid_height=8, magnitude=2)
    p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.sample(100)
    print("augemented")
    return