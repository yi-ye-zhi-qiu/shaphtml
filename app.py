import shap
from shap.plots._force_matplotlib import draw_additive_plot

from .models import give_shap_plot


app = Flask(__name__)

@app.route('/')

def displayshap():

    explainer, shap_values = give_shap_plot()

    def _force_plot_html(explainer, shap_values, ind):
        ind = list(ind)[0]
        force_plot = shap.plots.force(shap_values[ind], matplotlib=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        return shap_html

    shap_plots = {}

    for i in range(10):
        shap_plots[i] = _force_plot_html(explainer, shap_values, ind)
    return render_template('templates/displayshap.html', shap_plots = shap_plots)
