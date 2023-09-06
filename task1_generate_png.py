import random
import os
import json
from PIL import Image, ImageDraw
import math

class RandomShapeGenerator:
    def __init__(self, image_size=(256, 256), min_size=25, max_size=150):
        self.image_size = image_size
        self.min_size = min_size
        self.max_size = max_size
        self.shapes = ['rectangle', 'triangle', 'circle', 'hexagon']

    def generate_random_shape(self, existing_shapes):
        shape_type = random.choice(self.shapes)
        size = random.randint(self.min_size, self.max_size)
        
        # Генерируем начальные координаты так, чтобы фигура умещалась в пределах изображения
        x = random.randint(0, self.image_size[0] - size)
        y = random.randint(0, self.image_size[1] - size)
        
        color = self.get_figure_color()

        rotation_angle = random.randint(-30, 30)  # Ограничиваем поворот до -30 до 30 градусов

        new_shape = (shape_type, size, (x, y, x + size, y + size), color, rotation_angle)

        if not self.is_overlapping(new_shape, existing_shapes) and not self.is_outside_image(new_shape):
            return new_shape
        else:
            return self.generate_random_shape(existing_shapes)

    def is_overlapping(self, new_shape, existing_shapes):
        for shape in existing_shapes:
            if self.do_shapes_overlap(new_shape[2], shape[2]):
                return True
        return False

    def do_shapes_overlap(self, rect1, rect2):
        return not (rect1[2] < rect2[0] or
                    rect1[3] < rect2[1] or
                    rect1[0] > rect2[2] or
                    rect1[1] > rect2[3])

    def is_outside_image(self, shape):
        x1, y1, x2, y2 = shape[2]
        if x1 < 0 or y1 < 0 or x2 > self.image_size[0] or y2 > self.image_size[1]:
            return True
        return False
    
    def get_background_color(self):
        return random.randint(0, 127), random.randint(0, 127), random.randint(0, 127)

    def get_figure_color(self):
        return random.randint(127, 255), random.randint(127, 255), random.randint(127, 255)

    def create_image_with_shapes(self, num_shapes=1):
        img = Image.new('RGB', self.image_size, self.get_background_color())
        draw = ImageDraw.Draw(img)
        descriptions = []
        shapes = []

        for _ in range(num_shapes):
            shape = self.generate_random_shape(shapes)
            shape_type, size, position, color, rotation_angle = shape
            rotated_shape = self.rotate_shape(shape_type, size, position, color, rotation_angle)

            if shape_type == 'rectangle':
                draw.rectangle(rotated_shape[2], fill=rotated_shape[3], outline=rotated_shape[3])
                descriptions.append({'size': size, 'position': position[:2], 'shape_type': 'rectangle'})
            elif shape_type == 'triangle':
                triangle_points = self.get_triangle_points(rotated_shape[2], rotation_angle)
                draw.polygon(triangle_points, fill=rotated_shape[3], outline=rotated_shape[3])
                descriptions.append({'size': size, 'position': position[:2], 'shape_type': 'triangle'})
            elif shape_type == 'circle':
                draw.ellipse(rotated_shape[2], fill=rotated_shape[3], outline=rotated_shape[3])
                descriptions.append({'size': size, 'position': position[:2], 'shape_type': 'circle'})
            elif shape_type == 'hexagon':
                hexagon_points = self.get_hexagon_points(rotated_shape[2], rotation_angle)
                draw.polygon(hexagon_points, fill=rotated_shape[3], outline=rotated_shape[3])
                descriptions.append({'size': size, 'position': position[:2], 'shape_type': 'hexagon'})

            shapes.append(shape)

        # img.show()
        return descriptions, img

    def rotate_shape(self, shape_type, size, position, color, rotation_angle):
        shape_img = Image.new('RGBA', (size, size))
        draw = ImageDraw.Draw(shape_img)

        if shape_type == 'rectangle':
            draw.rectangle((0, 0, size, size), fill=color)
        elif shape_type == 'triangle':
            half_size = size / 2
            points = [(half_size, 0), (size, size), (0, size)]
            points = [(x - half_size, y - half_size) for x, y in points]
            rotated_points = [
                (
                    math.cos(math.radians(rotation_angle)) * x - math.sin(math.radians(rotation_angle)) * y,
                    math.sin(math.radians(rotation_angle)) * x + math.cos(math.radians(rotation_angle)) * y
                )
                for x, y in points
            ]
            rotated_points = [(x + half_size, y + half_size) for x, y in rotated_points]
            draw.polygon(rotated_points, fill=color)
        elif shape_type == 'circle':
            draw.ellipse((0, 0, size, size), fill=color)
        elif shape_type == 'hexagon':
            half_size = size / 2
            points = [
                (half_size, 0),
                (size, size / 3),
                (size, 2 * size / 3),
                (half_size, size),
                (0, 2 * size / 3),
                (0, size / 3)
            ]
            points = [(x - half_size, y - half_size) for x, y in points]
            rotated_points = [
                (
                    math.cos(math.radians(rotation_angle)) * x - math.sin(math.radians(rotation_angle)) * y,
                    math.sin(math.radians(rotation_angle)) * x + math.cos(math.radians(rotation_angle)) * y
                )
                for x, y in points
            ]
            rotated_points = [(x + half_size, y + half_size) for x, y in rotated_points]
            draw.polygon(rotated_points, fill=color)

        rotated_shape_img = shape_img.rotate(rotation_angle, expand=True)
        bbox = rotated_shape_img.getbbox()
        rotated_shape = (shape_type, bbox[2] - bbox[0], (position[0] + bbox[0], position[1] + bbox[1], position[0] + bbox[2], position[1] + bbox[3]), color)

        return rotated_shape

    def get_triangle_points(self, rect, rotation_angle):
        half_width = (rect[2] - rect[0]) / 2
        half_height = (rect[3] - rect[1]) / 2
        center_x = rect[0] + half_width
        center_y = rect[1] + half_height

        # Вычисляем вершины треугольника
        points = [
            (center_x, center_y - half_height),
            (center_x + half_width * math.cos(math.radians(60)), center_y + half_height * math.sin(math.radians(60))),
            (center_x - half_width * math.cos(math.radians(60)), center_y + half_height * math.sin(math.radians(60)))
        ]

        # Поворачиваем вершины
        rotated_points = [
            (
                math.cos(math.radians(rotation_angle)) * (x - center_x) - math.sin(math.radians(rotation_angle)) * (y - center_y) + center_x,
                math.sin(math.radians(rotation_angle)) * (x - center_x) + math.cos(math.radians(rotation_angle)) * (y - center_y) + center_y
            )
            for x, y in points
        ]

        return rotated_points

    def get_hexagon_points(self, rect, rotation_angle):
        half_width = (rect[2] - rect[0]) / 2
        half_height = (rect[3] - rect[1]) / 2
        center_x = rect[0] + half_width
        center_y = rect[1] + half_height

        # Вычисляем вершины шестиугольника
        points = []
        for i in range(6):
            x = center_x + half_width * math.cos(math.radians(60 * i))
            y = center_y + half_height * math.sin(math.radians(60 * i))
            points.append((x, y))

        # Поворачиваем вершины
        rotated_points = [
            (
                math.cos(math.radians(rotation_angle)) * (x - center_x) - math.sin(math.radians(rotation_angle)) * (y - center_y) + center_x,
                math.sin(math.radians(rotation_angle)) * (x - center_x) + math.cos(math.radians(rotation_angle)) * (y - center_y) + center_y
            )
            for x, y in points
        ]

        return rotated_points

if __name__ == '__main__':
    output_dir = 'generated_images'
    os.makedirs(output_dir, exist_ok=True)
    generator = RandomShapeGenerator()

    # Список для хранения описаний фигур
    shape_descriptions = []

    for i in range(100):
        num_shapes = random.randint(1, 5)
        descriptions, img = generator.create_image_with_shapes(num_shapes)

        img_filename = os.path.join(output_dir, f'{i + 1:03}.png')
        img.save(img_filename)

        # Создаем описание фигур для текущей фотографии
        image_shapes = []
        for j, figure_desc in enumerate(descriptions):
            shape_type = figure_desc['shape_type']  # Извлекаем тип фигуры (название)
            region_info = figure_desc['position']
            origin = {"x": str(region_info[0]), "y": str(region_info[1])}
            size = figure_desc['size']
            size_info = {"width": str(size + region_info[0]), "height": str(size + region_info[1])}
            shape_info = {
                "id": str(j + 1),
                "name": shape_type,
                "region": {
                    "origin": origin,
                    "size": size_info
                }
            }
        
            shape_descriptions.append(shape_info)

        # Сохраняем описания фигур в JSON файл
        json_filename = os.path.join(output_dir, f"{i + 1:03}.json")
        with open(json_filename, "w") as json_file:
            json.dump(shape_descriptions, json_file, indent=4)
