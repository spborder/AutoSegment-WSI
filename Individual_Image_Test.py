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

import torch
import torchvision

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

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

import wsi_annotations_kit as wak


def gen_layout():

    main_layout = html.Div([
        html.H1('Segment Anything Model (SAM): for Histology image annotation'),
        html.Hr(),
        html.B(),
        dbc.Row([
            dbc.Col([
                html.H3('Current Image'),
                html.B(),
                dcc.Graph(id='current-image',figure=go.Figure()),
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
             Input('predict-all','n_clicks')],
             prevent_initial_call=True
        )(self.predict)

    def predict(self,click_coords,butt_click):

        if ctx.triggered_id=='current-image':

            new_image = self.predict_from_point(click_coords)
        
        elif ctx.triggered_id =='predict-all':

            new_image = self.predict_all_regions()
        
        else:
            print(f'Bad triggered_id: {ctx.triggered_id}')
            raise exceptions.PreventUpdate

        return new_image


    def predict_from_point(self,click_coords):

        if not click_coords is None:
            point_coords = np.array([[click_coords['points'][0]['y'],click_coords['points'][0]['x']]])

            point_label = np.array([1])

            masks, scores, logits = self.predictor.predict(
                point_coords = point_coords,
                point_labels = point_label,
                multimask_output = True
            )

            print(np.shape(masks))
            print(f'scores: {scores}')
            print(f'logits: {logits}')

            return 
        
        else:
            raise exceptions.PreventUpdate
    
    def predict_all_regions(self):

        masks = self.mask_generator(self.current_image)

        print(f'shape of masks: {np.shape(masks)}')


        return 

            



def main():


    main_layout = gen_layout()
    main_app = DashProxy(__name__,external_stylesheets = [dbc.themes.LUX])

    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    model_type = 'vit_h'
    device = 'cpu'

    sam = sam_model_registry[model_type](checkpoint = sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)

    ex_image = np.array(Image.open('./examples/ex_img.png'))
    predictor.set_image(ex_image)


    sam_app = SingleImageTest(main_app,main_layout,predictor,mask_generator)


if __name__=='__main__':
    main()





























