
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.image as img
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.rdCoordGen import AddCoords

import io
import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_learning_curve(train_losses,val_losses):
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    plt.plot(train_losses, label='Training loss', c= 'k')
    plt.plot(val_losses, label='Validation loss', c= 'r')
    y_max = np.concatenate([train_losses, val_losses]).mean() * 3
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, y_max)
    plt.legend()
    plt.title('Learning curve')

    fig = plt.gcf()
    fig.canvas.draw()
    pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    return pil_image

def plot_prediction(targets, prediction_df):
    """
    @targets: list of target names
    @prediction_df: dataframe with columns ["set","{target}_true","{target}_pred"]
        columns "set" is the set name, "train","val" or "test"
    """
    train_df = prediction_df[prediction_df['set']=="train"]
    val_df = prediction_df[prediction_df['set']=="val"]
    test_df = prediction_df[prediction_df['set']=="test"]
    
    target_nums = len(targets)
    row_num = 1
    col_num = 1
    while True:
        if row_num*col_num>=target_nums:
            break
        if row_num==col_num:
            row_num+=1
        else:
            col_num+=1
    plt.subplots(row_num,col_num,figsize=(col_num*6,row_num*5))
    for i in range(target_nums):
        plt.subplot(row_num,col_num,i+1)
        plt.scatter(train_df[f"{targets[i]}_true"],train_df[f"{targets[i]}_pred"],label="train",c='k',s=0.5,alpha=0.3)
        plt.scatter(val_df[f"{targets[i]}_true"],val_df[f"{targets[i]}_pred"],label="val",c='r',s=0.5,alpha=0.8)
        plt.scatter(test_df[f"{targets[i]}_true"],test_df[f"{targets[i]}_pred"],label="test",c='b',s=0.5,alpha=0.8)
        plt.xlabel("target")
        plt.ylabel("prediction")
        plt.title(targets[i])
        plt.legend()

    fig = plt.gcf()
    fig.canvas.draw()
    pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    return pil_image    
    

    
def mol_with_atom_index( mol ):
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def showAtomHighlight(mol,atms_list,log=True,atom_with_index=True):
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
    atom_num = mol.GetNumAtoms()
    if atom_with_index:
        mol=mol_with_atom_index(mol)
    hit_ats=[]
    hit_bonds = []
    hit_ats_colormap={}
    hit_bonds_colormap={}
    
    colormap=[(1.0, 0.75, 0.79),
              (0.9, 0.85, 0.81),
               (0.95, 0.95, 0.65),
               (0.74, 0.99, 0.79),
               (0.68, 0.85, 0.90),
               (0.87, 0.63, 0.87),
               (0.90, 0.90, 0.98),
               (0.82, 0.94, 0.75),
               (0.56, 0.93, 0.56),
               (0.50, 1.0, 0.83),
               (0.80, 0.60, 1.0),
               (1.0, 0.85, 0.73),
               (0.87, 0.65, 0.95),
               (0.53, 0.81, 0.92),
               (1.0, 0.75, 0.79),
               (1.0, 0.94, 0.84)
              ]
    for count,atms in enumerate(atms_list):
        atms = [int(x) for x in atms]
        hit_ats=hit_ats+atms
        for i in range(len(atms)):
            hit_ats_colormap[atms[i]]=colormap[count%len(colormap)]
            for j in range(i+1,len(atms)):
                bond=mol.GetBondBetweenAtoms(atms[i],atms[j])
                if bond:
                    bond_index=bond.GetIdx()
                    hit_bonds.append(bond_index)
                    hit_bonds_colormap[bond_index]=colormap[count%len(colormap)]
                    
                    
    AllChem.Compute2DCoords(mol)
    
    AddCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(1000, 800)
    d.DrawMolecule(mol,highlightAtoms = hit_ats, highlightAtomColors=hit_ats_colormap,
                   highlightBonds=hit_bonds, highlightBondColors=hit_bonds_colormap)
    d.FinishDrawing()
    d = d.GetDrawingText()
    
    img_buf = io.BytesIO(d)
    plot(img_buf)
    
def drawSmiles(smiles):
    img_buf = io.BytesIO()
    img = Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(300,300))
    img.save(img_buf, format='png')
    img_buf.seek(0)
    plot(img_buf)

def plot(buf):
    image = PIL.Image.open(buf)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

import ipywidgets as widgets
from IPython.display import display
import os

def loader_ui(tunnel:dict):
    model_list = os.listdir("_Models")
    model_list.sort()
    model_list=['Select model path']+model_list
    dd_widget = widgets.Dropdown(
        options=model_list,
        disabled=False,
    )

    global current_path,pred_img,lc_img
    pred_img =widgets.Image(
        format='png',
        width=400,
        height=300,
    )
    lc_img =widgets.Image(
        format='png',
        width=400,
        height=300,
    )
    output = widgets.Output()
    box2 = widgets.HBox([pred_img,lc_img])
    box1 = widgets.VBox([dd_widget, box2])

    display(box1,output)

    current_path = None
    tunnel['current_path'] = current_path
    def on_change(change):
        global  current_path,pred_img,lc_img
        if type(change.new)==str and current_path!=change.new:
            current_path=change.new
            img1 =  open("_Models/"+current_path+"/prediction.png", "rb").read()
            img2 =  open("_Models/"+current_path+"/learning_curve.png", "rb").read()
            pred_img.value = img1
            lc_img.value= img2
            tunnel['current_path'] = current_path
        
    dd_widget.observe(on_change)