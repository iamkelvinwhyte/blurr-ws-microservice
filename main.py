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

def lens_blur(img, radius=3, components=5, exposure_gamma=5):

    img = np.ascontiguousarray(img.transpose(2,0,1), dtype=np.float32)


    # Obtain component parameters / scale values
    parameters, scale = get_parameters(component_count = components)

    # Create each component for size radius, using scale and other component parameters
    components = [complex_kernel_1d(radius, scale, component_params['a'], component_params['b']) for component_params in parameters]

    # Normalise all kernels together (the combination of all applied kernels in 2D must sum to 1)
    components = normalise_kernels(components, parameters)

    # Increase exposure to highlight bright spots
    img = np.power(img, exposure_gamma)

    # Process RGB channels for all components
    component_output = list()
    for component, component_params in zip(components, parameters):
        channels = list()
        for channel in range(img.shape[0]):
            inter = signal.convolve2d(img[channel], component, boundary='symm', mode='same')
            channels.append(signal.convolve2d(inter, component.transpose(), boundary='symm', mode='same'))

        # The final component output is a stack of RGB, with weighted sums of real and imaginary parts
        component_image = np.stack([weighted_sum(channel, component_params) for channel in channels])
        component_output.append(component_image)

    # Add all components together
    output_image = reduce(np.add, component_output)

    # Reverse exposure
    output_image = np.clip(output_image, 0, None)
    output_image = np.power(output_image, 1.0/exposure_gamma)

    # Avoid out of range values - generally this only occurs with small negatives
    # due to imperfect complex kernels
    output_image = np.clip(output_image, 0, 1)

    #output_image *= 255
    #output_image = output_image.transpose(1,2,0).astype(np.uint8)
    output_image = output_image.transpose(1,2,0)
    return output_image

def motion_blur(img, size=None, angle=None):
    '''Motion blur generator'''
    if size is None:
        size = randint(20, 80)
    if angle is None:
        angle = randint(15, 30)

    k = np.zeros((size, size), dtype=np.float32)
    k[(size-1)//2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size/2-0.5, size/2-0.5), angle, 1.0), (size, size))
    k = k * (1.0/np.sum(k))

    return cv2.filter2D(img, -1, k)

async def hello(websocket, path):
    name = await websocket.recv()
    print(f"< {name}")

    greeting = f"Hello {name}!"

    await websocket.send(greeting)
    print(f"> {greeting}")

start_server = websockets.serve(hello, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
