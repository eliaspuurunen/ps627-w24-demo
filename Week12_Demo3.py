import bokeh as bk
import bokeh.io as bio
import bokeh.plotting as bplt
import bokeh.models as bmod
import bokeh.palettes as bp
import pandas as pd
import statsmodels.formula.api as smf

smokingData = pd.read_csv('Table1_SmokingVsCancer.csv')

bk.io.output_file('index.html', title = 'Smoking vs Lung Cancer Data')

x = smokingData['Year']
ySmokers = smokingData['PercentSmokers']
yLungCancer = smokingData['LungCancerPer100']

plotTime = bk.plotting.figure(title = 'Smoking vs. Lung Cancer over Time',
                              x_axis_label = 'Year',
                              y_axis_label = '% Smokers',
                              y_range = (15, 40),
                              height = 500)

plotTime.extra_y_ranges['lungcancer'] = bmod.Range1d(start = 70, end = 150)

plotTime.add_layout(
    bmod.LinearAxis(
        axis_label = 'Lung Cancer Per 100,000',
        y_range_name = 'lungcancer'
        ), place = 'right')

plotTime.line(x, ySmokers,
              legend_label = '% Smokers',
              line_width = 2,
              color = bp.Spectral4[2])

plotTime.line(x, yLungCancer,
              legend_label = 'Lung Cancer Per 100,000',
              line_width = 2,
              y_range_name = 'lungcancer',
              color = bp.Spectral4[3])


regressionPlot = bplt.figure(
    title = 'Regression, does smoking predict lung cancer',
    x_axis_label = 'Smoking %',
    y_axis_label = 'Lung Cancer Per 100k',
    height = 500)

model = smf.ols('LungCancerPer100 ~ PercentSmokers', data = smokingData).fit()

lineOfFit = model.fittedvalues

summaryValues = model.get_prediction().summary_frame()

regressionPlot.circle(smokingData['PercentSmokers'],
            smokingData['LungCancerPer100'])

regressionPlot.line(smokingData['PercentSmokers'],
            lineOfFit,
            legend_label = 'Regression Line')

regressionPlot.line(smokingData['PercentSmokers'],
            summaryValues['obs_ci_lower'],
            legend_label = 'CI (Lower)',
            line_dash = 'dashed')

regressionPlot.line(smokingData['PercentSmokers'],
            summaryValues['obs_ci_upper'],
            legend_label = 'CI (Upper)',
            line_dash = 'dashed')

regressionResultTable = model.summary().tables[0].as_html()
regressionCoefTable = model.summary().tables[1].as_html()

regressionResultsTableHtml = regressionResultTable + regressionCoefTable

tableDiv = bmod.Div(text = regressionResultsTableHtml)

bk.plotting.save(
    bk.layouts.column(
        plotTime,
        bk.layouts.row(
            regressionPlot,
            tableDiv,
            sizing_mode = 'stretch_width'
        ),
        sizing_mode = 'stretch_width'))
