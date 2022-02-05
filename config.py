"""
Stores system configurations 
"""
import pyglet
from pyglet.image import Animation, AnimationFrame


window_height = 700
window_width = 1440

background = pyglet.image.load('assets/track3.png')
image_data = background.get_image_data()
image_data_width = image_data.width
pixels = image_data.get_data('RGBA', 4*image_data_width)
n_pixels = len(pixels)

EXPLOSION_ANIMATION_PERIOD = 0.07

explosion_images_image = pyglet.image.load('assets/explosion.png')
explosion_images = pyglet.image.ImageGrid(explosion_images_image, 2, 8)
explosion_images = explosion_images.get_texture_sequence()
for explosion_image in explosion_images:
    explosion_image.anchor_x = explosion_image.width//2
    explosion_image.anchor_y = explosion_image.height//2
    #center_anchor(explosion_image)
explosion_animation = \
    pyglet.image.Animation.from_image_sequence(explosion_images,
                                               EXPLOSION_ANIMATION_PERIOD,
                                               loop=False)




search_length = 40

class MovingSprite(pyglet.sprite.Sprite):
    def __init__(self, image, x, y, dx=0, dy=0, batch=None):
        super(MovingSprite, self).__init__(image, x, y, batch=batch)
        self.dx = dx
        self.dy = dy




