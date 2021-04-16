import numpy as np
import plotly.express as px
cpu_result = {
    "c5n.18xlarge":
    {'mlp':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [13.540891170501709, 12.476744413375854, 11.147228717803955, 17.353014707565308, 21.962525606155396],
          'inference time': [0.0021244105027646435, 0.0019159694107211366, 0.0020370142800467356, 0.01768301214490618, 0.031411869185311456]},
     'bert':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [1323.5947971343994, 1315.2153856754303, 1324.9581191539764, 1343.9357023239136, 1361.5333795547485],
          'inference time': [1.105727233448807, 1.0654324025523907, 1.0785843681315987, 1.1530808222537139, 1.1640879536161617]},
     "price": 4.428},
    "c5n.9xlarge" :
    {'mlp':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [11.189261436462402, 9.973739385604858, 8.801320791244507, 16.157137632369995, 21.00126004219055],
          'inference time': [0.0016932609129925163, 0.001653958340080417, 0.0017271272990168358, 0.0160597453311998, 0.03037759479211301]},
     'bert':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [1274.8850712776184, 1252.0746772289276, 1245.7388393878937, 1254.435753583908, 1267.532795906067],
          'inference time': [1.0891822272417497, 1.0944765854854972, 1.0928822792306239, 1.1291938995828434, 1.1484798740367501]},
     "price": 2.214},
    "c5n.4xlarge" :
    {'mlp':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [17.158438682556152, 13.947082281112671, 15.241694211959839, 20.412732362747192, 27.380159616470337],
          'inference time': [0.0019228640867739307, 0.0019206051923790757, 0.0019435675776734644, 0.01729791140069767, 0.032036843348522574]},
     'bert':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [1637.8760719299316, 1620.9459342956543, 1620.8361439704895, 1633.9754033088684, 1637.9916734695435],
          'inference time': [1.4184645949577799, 1.4048790737074248, 1.3930077564959624, 1.4117846683580049, 1.4288307124254656]},
     "price" : 0.984},

    "c5n.2xlarge" :
    {'mlp':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [14.361419200897217, 13.354588985443115, 13.591758489608765, 23.872464418411255, 37.43416905403137],
          'inference time': [0.002182399740024489, 0.0022328167545552155, 0.002277036102450624, 0.019505660144650206, 0.041018303559750925]},
     'bert':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [2447.5564823150635, 2455.2421414852142, 2459.3994550704956, 2476.5737640857697, 2503.035465478897],
          'inference time': [1.9077403946798674, 1.9332553440210771, 1.948980034614096, 2.001873848389606, 2.013627045008601]},
     "price" : 0.492},

    "m5.2xlarge" :
    {'mlp':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [21.01449203491211, 14.700647830963135, 15.15757966041565, 27.494479656219482, 39.67811942100525],
          'inference time': [0.002862286810972253, 0.002527393856827094, 0.0025834514170276876, 0.021360832817700445, 0.03851592784025231]},
     'bert':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [2731.6758856773376, 2757.0264995098114, 2768.441976070404, 2793.7750267982483, 2828.0964863300323],
          'inference time': [2.1820361796690495, 2.1762729554760214, 2.198035161105954, 2.296724497055521, 2.3419864579122893]},
     "price" : 0.46},
    "m5.4xlarge":
    {'mlp':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [12.471782684326172, 10.90599536895752, 10.597107887268066, 19.06341576576233, 27.280228853225708],
          'inference time': [0.0020289056155146385, 0.0020649542613905302, 0.0020645114840293416, 0.018404644362780512, 0.03332033084363353]},
     'bert':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [1813.6761920452118, 1794.9994223117828, 1794.1460082530975, 1826.1661627292633, 1843.6001889705658],
          'inference time': [1.5287325637681144, 1.5295593957511746, 1.522513641386616, 1.5916012574215324, 1.6103330205897897]},
     "price": 0.92},
    "m5.8xlarge":
        {'mlp':
             {'no classes': [2, 50, 100, 10000, 20000],
              'training time': [10.512910604476929, 9.75018835067749, 9.38976502418518, 15.95421051979065, 21.958120584487915],
              'inference time': [0.001863421226034359, 0.0018840760600810148, 0.0019181687004712162, 0.017100306189790065, 0.032234095797246815]},
         'bert':
             {'no classes': [2, 50, 100, 10000, 20000],
              'training time': [1418.350023984909, 1392.7381534576416, 1392.2020711898804, 1406.4629576206207, 1411.3244981765747],
              'inference time': [1.237773832009763, 1.220512145636033, 1.2142973785497704, 1.2685545622086039, 1.2907001303166759]},
         "price": 1.84},
    "m5.12xlarge":
        {'mlp':
             {'no classes': [2, 50, 100, 10000, 20000],
              'training time': [9.613126993179321, 10.028851747512817, 9.355329275131226, 15.407365560531616, 20.388332843780518],
              'inference time': [0.001749417003320188, 0.0017537936872365524, 0.0017739400571706344, 0.016351620761715636, 0.030961886960632946]},
         'bert':
             {'no classes': [2, 50, 100, 10000, 20000],
              'training time': [1266.4482853412628, 1255.6161534786224, 1258.1703221797943, 1271.8952269554138, 1285.798241853714],
              'inference time': [1.0944540257356605, 1.0703771199498857, 1.079936146736145, 1.1171560068519748, 1.1532167816648677]},
         "price": 2.76},

"g3.4xlarge":
        {'mlp':
             {'no classes': [2, 50, 100, 10000, 20000],
              'training time': [32.5131995677948, 19.993014574050903, 19.49515128135681, 24.664000511169434, 30.96623969078064],
              'inference time': [0.004747893129076276, 0.003015046217003647, 0.0029536595149916045, 0.023322056750861967, 0.04413742556863902]},
         'bert': {'no classes': [2, 50, 100, 10000, 20000],
                  'training time': [291.1605463027954, 282.62183809280396, 283.8296422958374, 284.0868787765503, 290.077486038208],
                  'inference time': [0.2634515287924786, 0.2599753190060051, 0.25928513249572444, 0.2771172231557418, 0.2983498244869466]},
         "price": 1.425 },
"g4dn.2xlarge":
    {'mlp':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [18.047739505767822, 16.740589380264282, 16.48866629600525, 21.18340301513672, 24.423452138900757],
          'inference time': [0.0027235089516153142, 0.0026681289380910446, 0.002651785101209368, 0.019234569705262477, 0.03599388867008443]},
     'bert':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [289.02646255493164, 297.60623145103455, 296.0161476135254, 299.6972622871399, 304.22694396972656],
          'inference time': [0.26689156099241607, 0.2696997730099425, 0.268976198167217, 0.2827193955985867, 0.30081511395318167]},
        "price": 0.94
     },

"p2.large":
    {'mlp':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [22.68250060081482, 21.693998098373413, 21.162556648254395, 27.44784450531006, 36.32786679267883],
          'inference time': [0.003396384570063377, 0.003343067607101129, 0.0033055738526947646, 0.02271166383003702, 0.042332296468773664]},
     'bert': {'no classes': [2, 50, 100, 10000, 20000],
              'training time': [497.88060331344604, 500.3348813056946, 503.30378127098083, 509.1608729362488, 510.04041266441345],
              'inference time': [0.45726126918987353, 0.49957541665252375, 0.454925858244604, 0.4603731084843071, 0.4907277056149074]},
     "price" : 1.326},

"p3.2xlarge":
    {'mlp':
         {'no classes': [2, 50, 100, 10000, 20000],
          'training time': [31.207317352294922, 17.045449256896973, 17.23259949684143, 20.84071135520935, 24.6668918132782],
          'inference time': [0.006852999025461625, 0.0026244095393589567, 0.0026388253484453473, 0.02073337593857123, 0.040009379386901855]},
     'bert': {'no classes': [2, 50, 100, 10000, 20000],
              'training time': [159.59198307991028, 144.85900735855103, 142.45676064491272, 149.44550561904907, 153.42596101760864],
              'inference time': [0.16683880893551573, 0.1748491002588856, 0.17342416850887998, 0.18658071026510123, 0.2051472955820512]},
     "price" : 3.823},
}

import plotly.graph_objects as go

def _get_color(n: int) -> str:
    color_lst = px.colors.qualitative.Set1
    return color_lst[n % len(color_lst)]

def plot(cpu_result, model_name, method, show_price, title, x_axis, y_axis):

    fig = go.Figure()
    for n, cpu_name in enumerate(["c5n.2xlarge", "c5n.4xlarge","c5n.9xlarge","c5n.18xlarge"]):
        item = cpu_result[cpu_name][model_name]
        no_x = len(item[method])
        y = np.array(item[method])
        if show_price:
            y = y * cpu_result[cpu_name]["price"] / (60 * 60)
        x = y * cpu_result[cpu_name]["price"] / (60 * 60)
        fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       name=cpu_name,
                       mode="markers",
                       marker=dict(size=[8+n*3] * no_x,
                                   color=[_get_color(1)] * no_x)))


    for n, cpu_name in enumerate(["m5.2xlarge", "m5.4xlarge", "m5.8xlarge", "m5.12xlarge"]):
        item = cpu_result[cpu_name][model_name]
        y = np.array(item[method])
        if show_price:
            y = y * cpu_result[cpu_name]["price"] / (60 * 60)
        fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       name=cpu_name,
                       mode="markers",
                       marker=dict(size=[8+n*3] * no_x,
                                   color=[_get_color(2)] * no_x)))
        #hovertemplate=   '<i>id</i>: %{y}' +'<b>%{text}</b>',text=text))

    for n, cpu_name in enumerate(["g3.4xlarge", "p2.large", "g4dn.2xlarge", "p3.2xlarge"]):
        item = cpu_result[cpu_name][model_name]
        y = np.array(item[method])
        if show_price:
            y = y * cpu_result[cpu_name]["price"] / (60 * 60)
        fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       name=cpu_name,
                       mode="markers",
                       marker=dict(
                           size=[8+n*3] * no_x,
                           color=[_get_color(0)] * no_x
                       )))

    fig.update_xaxes(title=x_axis)
    fig.update_yaxes(title=y_axis)
    fig.update_layout(
        title_text=title,
        width=500, height=500,
        xaxis = dict(
            tickmode='array',
            tickvals=x,
            ticktext=item["no classes"]))
    fig.show()
    filename = title.replace(' ', '_').replace('[', '_').replace("]", "_")
    fig.write_html(file=f"{filename}.html", auto_open=False)

def plot_scatter(cpu_result, model_name, method, title, x_axis, y_axis):

    fig = go.Figure()
    for n, cpu_name in enumerate(["c5n.2xlarge", "c5n.4xlarge","c5n.9xlarge","c5n.18xlarge"]):
        item = cpu_result[cpu_name][model_name]
        no_x = len(item[method])
        y = np.array(item[method])
        x = y * cpu_result[cpu_name]["price"] / (60 * 60)
        fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       name=cpu_name,
                       mode="markers",
                       marker=dict(size=[8+n*3] * no_x,
                                   color=[_get_color(1)] * no_x)))


    for n, cpu_name in enumerate(["m5.2xlarge", "m5.4xlarge", "m5.8xlarge", "m5.12xlarge"]):
        item = cpu_result[cpu_name][model_name]
        y = np.array(item[method])
        x = y * cpu_result[cpu_name]["price"] / (60 * 60)
        fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       name=cpu_name,
                       mode="markers",
                       marker=dict(size=[8+n*3] * no_x,
                                   color=[_get_color(2)] * no_x)))
        #hovertemplate=   '<i>id</i>: %{y}' +'<b>%{text}</b>',text=text))

    for n, cpu_name in enumerate(["g3.4xlarge", "p2.large", "g4dn.2xlarge", "p3.2xlarge"]):
        item = cpu_result[cpu_name][model_name]
        y = np.array(item[method])
        x = y * cpu_result[cpu_name]["price"] / (60 * 60)
        fig.add_trace(
            go.Scatter(x=x,
                       y=y,
                       name=cpu_name,
                       mode="markers",
                       marker=dict(
                           size=[8+n*3] * no_x,
                           color=[_get_color(0)] * no_x
                       )))

    fig.update_xaxes(title=x_axis)
    fig.update_yaxes(title=y_axis)
    fig.update_layout(
        title_text=title,
        width=600, height=600)
    fig.show()
    filename = title.replace(' ', '_').replace('[', '_').replace("]", "_")
    fig.write_html(file=f"{filename}.html", auto_open=False)

plot(cpu_result, "mlp", show_price=False, y_axis="runtime [sec]", x_axis="no. classes", method="training time",
     title = "Training Multilayer Perceptron Model [Runtime]")
plot(cpu_result, "bert", show_price=False, y_axis="runtime [sec]", x_axis="no. classes", method="training time",
     title = "Training BERT Model [Runtime]")
plot(cpu_result, "mlp", show_price=True, y_axis="price [usd]", x_axis="no. classes", method="training time",
     title = "Training Multilayer Perceptron Model [Price]")
plot(cpu_result, "bert", show_price=True, y_axis="price [usd]", x_axis="no. classes", method="training time",
     title="Training BERT Model [Price]")

plot(cpu_result, "mlp", show_price=False, y_axis="runtime [sec]", x_axis="no. classes", method="inference time",
     title = "Inference Multilayer Perceptron Model [Runtime]")
plot(cpu_result, "bert", show_price=False, y_axis="runtime [sec]", x_axis="no. classes", method="met",
     title = "Inference BERT Model [Runtime]")
plot(cpu_result, "mlp", show_price=True, y_axis="price [usd]", x_axis="no. classes", method="inference time",
     title = "Inference Multilayer Perceptron Model [Price]")
plot(cpu_result, "bert", show_price=True, y_axis="price [usd]", x_axis="no. classes", method="inference time",
     title="Inference BERT Model [Price]")


plot_scatter(cpu_result, "mlp", y_axis="runtime [sec]", x_axis="costs [usd]", method="training time",
     title = "Training Multilayer Perceptron Model [Runtime vs Costs]")
plot_scatter(cpu_result, "bert", y_axis="runtime [sec]", x_axis="costs [usd]", method="training time",
     title = "Training BERT Model [Runtime vs Costs]")

plot_scatter(cpu_result, "mlp", y_axis="runtime [sec]", x_axis="costs [usd]", method="inference time",
     title = "Inference Multilayer Perceptron Model [Runtime vs Costs]")
plot_scatter(cpu_result, "bert", y_axis="runtime [sec]", x_axis="costs [usd]", method="inference time",
     title = "Inference BERT Model [Runtime vs Costs]")
