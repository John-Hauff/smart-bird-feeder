import requests as req

def post_img():
    img_path = './captured-bird-images/bird_memory.jpeg' # path is constant (img overwritten)
    url = 'https://smart-bird-feeder-api.herokuapp.com/user/post-bird-memory'
    # url = 'http://localhost:3000/user/post-bird-memory'

    # Include the bird species in the filename, then parse the filename for the species in the express app
    files = {'file': ('bird_img.jpeg', open(
        img_path, 'rb'), 'image/jpeg')}
    r = req.post(url, files=files)
    print(r.text)
