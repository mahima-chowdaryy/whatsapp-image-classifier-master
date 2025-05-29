from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_image(filename, text, size=(800, 600), bg_color=(41, 45, 62), text_color=(255, 255, 255)):
    # Create a new image with the specified background color
    image = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(image)
    
    # Add some decorative elements
    draw.rectangle([(50, 50), (size[0]-50, size[1]-50)], outline=text_color, width=2)
    draw.rectangle([(100, 100), (size[0]-100, size[1]-100)], outline=text_color, width=1)
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position to center it
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, font=font, fill=text_color)
    
    # Save the image
    image.save(filename)

def main():
    # Create the gallery directory if it doesn't exist
    os.makedirs('static/images/gallery', exist_ok=True)
    
    # Generate generic images
    create_sample_image('static/images/gallery/generic1.jpg', 'Nature Scene')
    create_sample_image('static/images/gallery/generic2.jpg', 'City View')
    create_sample_image('static/images/gallery/generic3.jpg', 'Beach Sunset')
    
    # Generate non-generic images
    create_sample_image('static/images/gallery/non-generic1.jpg', 'Artistic Portrait')
    create_sample_image('static/images/gallery/non-generic2.jpg', 'Abstract Art')
    create_sample_image('static/images/gallery/non-generic3.jpg', 'Creative Design')
    
    # Generate document images
    create_sample_image('static/images/gallery/document1.jpg', 'Business Document')
    create_sample_image('static/images/gallery/document2.jpg', 'Legal Document')
    create_sample_image('static/images/gallery/document3.jpg', 'Medical Report')

if __name__ == '__main__':
    main() 