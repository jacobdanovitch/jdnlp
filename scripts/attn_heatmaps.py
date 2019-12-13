import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def run_example(sent=""):
    pred = model.predict(sent)
    
    # inst = model._json_to_instance({"sentence": "Test test test"})     
    # words = inst.fields['tokens'].tokens
    return pred


def attn_heatmaps(read_attn, ctrl_attn, cm=sns.light_palette("navy")):
    #read_attn = torch.cat(read_attn).T
    read_attn = read_attn.T
    # read_attn[[*range(5), *(-i for i in reversed(range(15)))]]
    #hm = sns.heatmap(read_attn.cpu().numpy(), cmap=cm)
    hm = sns.heatmap(read_attn, cmap=cm)
    
    # hm = sns.heatmap(read_attn[[*range(5), *(-i for i in reversed(range(15)))]].cpu().numpy(), cmap=cm)
    ticks = hm.yaxis.get_major_ticks()
    hm.set_yticklabels(['topic_{}'.format('n' if (i == len(ticks)-1) else i) for i in range(len(ticks))], rotation=(360))
    for t in ticks[2:-1]: 
        t.set_visible(False)
    
    # plt.savefig('visuals/mac-read-heatmap.png'); plt.clf()
    
    #ctrl_attn = torch.cat(ctrl_attn).T
    ctrl_attn = ctrl_attn.T
    # hm = sns.heatmap(ctrl_attn.T.cpu().numpy(), cmap=cm)
    hm = sns.heatmap(ctrl_attn, cmap=cm)
    hm.set_yticklabels(['w_{}'.format(i) for (i, w) in enumerate(words)], rotation=(45))
    # plt.savefig('visuals/mac-control-heatmap.png'); plt.clf()

    plt.show()
    
    
def generate_html(text_list, attention_list, latex_file, color='red', rescale_value = False):
    assert(len(text_list) == len(attention_list))
    if rescale_value:
        attention_list = rescale(attention_list)
    word_num = len(text_list)
    text_list = clean_word(text_list)
    string = ''
    for idx in range(word_num):
        string += f"<p style='background-color: rgba({color}, {attention_list[idx]})'>{text_list[idx]}</p>"
    return string


latex_special_token = ["!@#$%^&*()"]

def unwrap_tensor(t):
    return t.cpu().numpy()

def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale# .tolist()

def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list

def attn_latex(f, text_list, attention_list, color, rescale_value):
    if(len(text_list) < len(attention_list)):
        assert sum(attention_list[len(text_list):]) < 0.25
        # text_list += ['[PAD]'] * (len(attention_list) - len(text_list))
    if rescale_value:
        attention_list = rescale(attention_list)

    text_list = clean_word(text_list)
    
    string = r'''\begin{CJK*}{UTF8}{gbsn}''' #+ "\n" 
    string += r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''#+"\n"
    for word, weight in zip(text_list, attention_list):
        string += "\\colorbox{%s!%s}{"%(color, weight)+"\\strut " + word+"} "
    string += "\n}}}"
    string += "\n\end{CJK*}"
    return string

def generate(text_list, attention_list, latex_file, color='red', rescale_value = False):
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth]{standalone}
    \special{papersize=210mm,297mm}
    \usepackage{color}
    \usepackage{tcolorbox}
    \usepackage{CJK}
    \usepackage{adjustbox}
    \tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
    \begin{document}'''+'\n')
        string = ''
        if isinstance(color, list):
            for t, a, c in zip(text_list, attention_list, color):
                string += attn_latex(f, t, a, c, rescale_value)
        else:
            string += attn_latex(f, text_list, attention_list, color, rescale_value)
        f.write(string)
        f.write(r'''\end{document}''')
        
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")