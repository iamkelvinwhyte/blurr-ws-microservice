import asyncio
import websockets
from PIL import Image, ImageFilter
import cv2
def blur (image_data):
    image = cv2.imread(image_data) # reads the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to HSV
    figure_size = 9 # the dimension of the x and y axis of the kernal.
    new_image = cv2.blur(image,(figure_size, figure_size))
    return new_image

def greyscale (image_data):
    image = cv2.imread(image_data)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    figure_size = 9
    new_image = cv2.blur(image,(figure_size, figure_size))
    return new_image

def gaussian_blur (image_data):
    image = cv2.imread(image_data)
    new_image = cv2.GaussianBlur(image, (figure_size, figure_size),0)
    return new_image

def median_blur (image_data):
    image = cv2.imread(image_data)
    new_image = cv2.medianBlur(image, figure_size)
    return new_image

def frequency_blur (image_data):
    image = cv2.imread(image_data)
    dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT)
    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    return dft


async def hello(websocket, path):
    name = await websocket.recv()
    print(f"< {name}")

    greeting = f"Hello {name}!"

    await websocket.send(greeting)
    print(f"> {greeting}")

start_server = websockets.serve(hello, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
