import io
from PIL import Image

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img