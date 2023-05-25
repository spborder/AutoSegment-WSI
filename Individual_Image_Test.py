"""

Making some sort of interactive Segment Anything Model (SAM) for Whole Slide Image annotation correction

"""


import os
import sys
import numpy as np
import lxml.etree as ET
from geojson import Feature, dump 
import json
import geojson

from tqdm import tqdm

import torch
import torchvision

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide

from PIL import Image

import plotly.express as px
import plotly.graph_objects as go

from dash import dcc, ctx, exceptions
import dash_bootstrap_components as dbc
import dash_leaflet as dl

from dash_extensions.enrich import html, DashProxy
from dash_extensions.javascript import arrow_function

from dataclasses import dataclass, field
from typing import Callable, List, Union
from dash.dependencies import handle_callback_args
from dash.dependencies import Input, Output, State

from wsi_annotations_kit import wsi_annotations_kit as wak


def gen_layout(image):

    main_layout = html.Div([
        html.H1('Segment Anything Model (SAM): for Histology image annotation'),
        html.Hr(),
        html.B(),
        dbc.Row([
            dbc.Col([
                html.H3('Current Image'),
                html.B(),
                dcc.Graph(id='current-image',figure=go.Figure(px.imshow(image))),
                dbc.Row([
                    dbc.Col(html.Div(
                        dbc.Button(
                            'Clear Masks',
                            id = 'clear-butt'
                        )
                    ),md=4),
                    dbc.Col(html.Div(
                        dbc.Button(
                            'Save Mask',
                            id = 'save-butt'
                        )
                    ),md=8)
                ])
            ],md=8),

            dbc.Col([
                html.H3('Annotation Options'),
                html.Hr(),
                html.B(),
                dbc.Card(
                    id = 'ann-options',
                    children = [
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Button(
                                    'Predict All',
                                    id = 'predict-all'
                                )
                            ])
                        ])
                    ]
                )
            ])
        ])
    ])

    return main_layout



class SingleImageTest:
    def __init__(self,
                 app,
                 layout,
                 predictor,
                 mask_generator,
                 current_image):
    
        self.app = app
        self.app.layout = layout
        self.predictor = predictor
        self.mask_generator = mask_generator
        self.current_image = current_image

        self.app.callback(
            Output('current-image','figure'),
            [Input('current-image','clickData'),
             Input('predict-all','n_clicks'),
             Input('clear-butt','n_clicks')],
             prevent_initial_call=True
        )(self.predict)


        self.app.run_server(host = '0.0.0.0',debug=False,use_reloader=False,port=8050)

    def predict(self,click_coords,butt_click,clear_click):

        if ctx.triggered_id=='current-image':

            new_image = self.predict_from_point(click_coords)
        
        elif ctx.triggered_id =='predict-all':

            new_image = self.predict_all_regions()

        elif ctx.triggered_id =='clear-butt':
            new_image = self.current_image
        
        else:
            print(f'Bad triggered_id: {ctx.triggered_id}')
            raise exceptions.PreventUpdate

        return px.imshow(new_image)


    def predict_from_point(self,click_coords):

        if not click_coords is None:
            # Points for input into SAM have to be (x,y)
            point_coords = np.array([[click_coords['points'][0]['x'],click_coords['points'][0]['y']]])

            point_label = np.array([1])

            masks, scores, logits = self.predictor.predict(
                point_coords = point_coords,
                point_labels = point_label,
                multimask_output = True
            )

            highest_score_mask = masks[np.argmax(scores),:,:][:,:,None]

            # Creating mask overlay using PIL
            color_mask_4d = self.generate_mask(highest_score_mask)
            overlaid_mask = Image.fromarray(np.uint8(self.current_image)).convert('RGBA')
            overlaid_mask.paste(color_mask_4d,mask=color_mask_4d)

            return overlaid_mask
        
        else:
            raise exceptions.PreventUpdate
    
    def predict_all_regions(self):

        masks = self.mask_generator.generate(self.current_image)

        print(f'shape of masks: {np.shape(masks)}')
        if not len(masks) == 0:

            sorted_anns = sorted(masks,key = (lambda x: x['area']), reverse = True)
            final_mask = np.uint8(np.zeros((sorted_anns[0]['segmentation'].shape[0],sorted_anns[0]['segmentation'].shape[1],4)))

            #print(f'shape of final mask: {np.shape(final_mask)}')
            for ann in tqdm(sorted_anns):
                
                ann_mask = self.generate_mask(ann['segmentation'][:,:,None])
                #print(f'shape of ann_mask: {np.shape(np.array(ann_mask))}')
                final_mask+=np.array(ann_mask)

            final_mask = Image.fromarray(final_mask)
            overlaid_mask = Image.fromarray(np.uint8(self.current_image)).convert('RGBA')
            overlaid_mask.paste(final_mask,mask=final_mask)

            return overlaid_mask
        else:

            print(f'masks shape is zero')
            return self.current_image
        

    def generate_mask(self,mask):
        zero_mask = np.where(np.sum(mask,axis=-1)==0,0,128)[:,:,None]
        
        random_color = [np.random.randint(0,255) for i in range(3)]
        
        color_mask = np.concatenate((random_color[0]*mask,random_color[1]*mask,random_color[2]*mask),axis=-1)
        color_mask_4d = Image.fromarray(np.uint8(np.concatenate((color_mask,zero_mask),axis=-1)),'RGBA')

        return color_mask_4d






def main():


    sam_checkpoint = './models/sam_vit_h_4b8939.pth'
    model_type = 'vit_h'
    device = 'cpu'

    sam = sam_model_registry[model_type](checkpoint = sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)

    ex_image = np.uint8(np.array(Image.open('./examples/ex_img.PNG')))[:,:,0:3]
    print(f'shape of ex_image: {np.shape(ex_image)}')
    # Getting it into the size that SAM expects
    #ex_image = ResizeLongestSide(1024).apply_image(ex_image)
    #print(f'Shape of processed image: {np.shape(ex_image)}')
    #ex_image = np.moveaxis(ex_image,source=-1,destination=0)[None,:,:,:]
    #print(f'Shape of batched and channels first image: {np.shape(ex_image)}')
    predictor.set_image(ex_image)


    main_layout = gen_layout(ex_image)
    main_app = DashProxy(__name__,external_stylesheets = [dbc.themes.LUX])


    sam_app = SingleImageTest(main_app,main_layout,predictor,mask_generator,ex_image)


if __name__=='__main__':
    main()





























