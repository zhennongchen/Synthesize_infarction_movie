from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data import DataSet
import os.path
import os
import settings
import numpy as np
import function_list as ff
cg = settings.Experiment() 
main_folder = os.path.join(cg.oct_main_dir)

def validate(data_type, model,excel_save_path,learning_rate,learning_decay,seq_length=40, saved_model=None,
             class_limit=None, image_shape=None):
    
    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    if model == 'lstm_regression':
        regression = 1
        sequence_len = 2 # for researchmodel
    else:
        regression = 0
        sequence_len = seq_length
    

    train,test = data.split_train_test()
    rm = ResearchModels(len(data.classes), model, sequence_len, learning_rate,learning_decay,saved_model)

    final_result_list = ()
    for sample in train:
        movie_id = sample[2]
        if movie_id.split('_')[2] == '277235': # exclude from validation 
            print(movie_id)
            continue
        p_generator = data.predict_generator(sample, data_type,regression)
        predict_output = rm.model.predict_generator(generator=p_generator,steps = 1)
        
        if regression == 0:
            if sample[1] == 'normal':
                truth = 0
            elif sample[1] == 'mild':
                truth = 1
            else:
                truth = 2

            if np.argmax(predict_output[0]) == 0: # mild = [1,0,0], normal=[0,1,0],severe = [0,0,1]
                predict = 1
            elif np.argmax(predict_output[0]) == 1:
                predict = 0
            else:
                predict = 2
        else:
            truth = float(sample[2].split('_')[-1])
            predict = predict_output[0][0]

        result = [movie_id,truth,predict]
        final_result_list= (*final_result_list, result)
        
    
    par = [('movie_ID',1),('truth',1),('predict',1)]
    ff.xlsx_save(excel_save_path,final_result_list,par,list(range(0,3)))
    
        
         
    
def main():
    model = 'lstm'
    rank = '4'
    saved_model = os.path.join(main_folder,'checkpoints',model+rank,'lstm-127.hdf5')
    seq_len = 20
    learning_rate = 1e-4
    learning_decay = 1e-5

    # create folder to save result as excel file
    excel_folder = os.path.join(main_folder,'validation_result')
    ff.make_folder([excel_folder])
    excel_save_path = os.path.join(excel_folder,model+rank+'_trainingdata.xlsx')

    if model == 'conv_3d' or model == 'lrcn':
        data_type = 'images'
        image_shape = (80, 80, 3)
    else:
        data_type = 'features'
        image_shape = None

    validate(data_type, model,excel_save_path, learning_rate,learning_decay, seq_length = seq_len,saved_model=saved_model,
             image_shape=image_shape, class_limit=None)

if __name__ == '__main__':
    main()
