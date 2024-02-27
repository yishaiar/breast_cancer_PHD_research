import matplotlib.pyplot as plt
def backgroundColor():
    return 'gainsboro' #gainsboro,lightgrey,whitesmoke
def hex_to_rgba(hex):
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        # normalized rgb
        rgb.append(decimal/255)
    
    # alpha
    rgb.append(0.5)
    return tuple(rgb)

def cluster_colors():
    

    hex = ['e6194b', '3cb44b', 'ffe119', '4363d8', 'f58231', '911eb4', '46f0f0', 'f032e6', 
        'bcf60c', 'fabebe', '008080', 'e6beff', '9a6324', 'fffac8', '800000', 'aaffc3',
            '808000', 'ffd8b1', '000075', '808080', ]
    return [hex_to_rgba(h) for h in hex]
def sample_colors():
    
    hex =  ['a6cee3','1f78b4','b2df8a','33a02c','fb9a99','e31a1c','fdbf6f','ff7f00','cab2d6','6a3d9a','ffff99','b15928']
    return [hex_to_rgba(h) for h in hex]

def class_colors():
    # unique classes: ['Basal-like', 'Cycling', 'Luminal', 'Noise', 'Unknown']
    hex =  ['fb9a99','3f78c1','33a02c',     '6a3d9a','ffff99',]
    return [hex_to_rgba(h) for h in hex]
def MeanDist(class1_data,class2_data,features,title='',font_size = 10):
    
    # sns.set_style({'legend.frameon':True})
 
    dd0=class1_data[features].mean().sort_values(ascending=False)
    dd1=class2_data[features].mean().sort_values()
    diffs=(dd1-dd0).sort_values(ascending=False)    
    
    clr=['darkgreen','purple']
    colors = [clr[0] if x < 0 else clr[1] for x in diffs]
    
    # fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    plt.figure(figsize=(6, 5))
    plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)
    # Decorations
    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=font_size  ) 
    plt.yticks(fontsize=font_size ) 
    # plt.xscale('symlog')
    

    plt.title(title, fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    
