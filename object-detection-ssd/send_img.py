from os import name
import requests as req


def post_bird_memory(species_name):
    # path is constant (img overwritten)
    img_path = './captured-bird-images/bird_memory.jpeg'
    # url = 'https://smart-bird-feeder-api.herokuapp.com/user/post-bird-memory'
    url = 'http://192.168.0.17:3000/user/post-bird-memory'

    files = {'file': (species_name, open(
    img_path, 'rb'), 'image/jpeg')}
    r = req.post(url, files=files)
    print(r.text)


if __name__ == '__main__':
    post_bird_memory("cardinal")
