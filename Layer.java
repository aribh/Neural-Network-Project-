package com.arib.neuralnetwork;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
public class Layer {
    public static DenseLayer dense(int inputs, int outputs, Activation activation){ 
        return new DenseLayer.Builder() 
        .nIn(inputs) 
        .nOut(outputs) 
        .activation(activation) 
        .build(); 
    }
    public static OutputLayer output(int inputs, int outputs){ 
        return new OutputLayer.Builder(LossFunctions.LossFunction.MSE) 
        .nIn(inputs) 
        .nOut(outputs) 
        .activation(Activation.IDENTITY) .build(); 
    }
}
