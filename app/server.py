import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.dropbox.com/s/3x66dm6h52ynz8b/fullfood.pkl?raw=1'
export_file_name = 'fullfood.pkl'

classes = ['apple_pie','baby_back_ribs','baklava', 'beef_carpaccio',
           'beef_tartare', 'beet_salad', 'beignets', 'bibimbap',
           'bread_pudding', 'breakfast_burrito', 'bruschetta',
           'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
           'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry',
           'chicken_quesadilla', 'chicken_wings', 'chocolate_cake',
           'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich',
           'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
           'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
           'escargots', 'falafel', 'filet_mignon', 'fish_and_chips',
           'foie_gras', 'french_fries', 'french_onion_soup',
           'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
           'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
           'grilled_salmon', 'guacamole', 'gyoza', 'hamburger',
           'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus',
           'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich',
           'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels',
           'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai',
           'paella', 'pancakes', 'panna_cotta', 'peking_ducks',
           'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
           'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
           'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
           'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara',
           'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi',
           'tacos', 'takoyako', 'tiramisu', 'tuna_tartare', 'waffles']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
