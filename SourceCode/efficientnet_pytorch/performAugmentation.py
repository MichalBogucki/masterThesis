import Augmentor


def performAugmentation():
    #augImageDirectory = "jpgImages/aug/test"
    augImageDirectory = "jpgImages/aug/moana"
    p = Augmentor.Pipeline(augImageDirectory)
    p.skew(probability=0.5)
    p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
    p.random_distortion(probability=0.5, grid_width=8, grid_height=8, magnitude=2)
    p.random_erasing(probability=0.4, rectangle_area=0.4)
    p.random_brightness(probability=0.3,min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=0.3,min_factor=0.8, max_factor=1.2)
    p.random_color(probability=0.3,min_factor=0.8, max_factor=1.2)
    p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.1)
    p.zoom_random(probability=0.4, percentage_area=0.85)
    p.sample(300)
    print("augemented")
    return
