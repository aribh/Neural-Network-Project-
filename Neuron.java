package com.arib.neuralnetwork;

public class Neuron {
    private double [] weights;
    private double bias;
    private Activation activation;

    public Neuron(double [] weights, double bias, Activation activation){
        this.weights = weights;
        this.bias = bias;
        this.activation = activation;
    }
    public double computeZ(double [] inputs){
        double z = bias; 
        for(int i = 0; i < inputs.length;i++){
            z += weights[i] * inputs[i];
        }
        return z;
    }
    public double forward(double[] inputs){
        double z = computeZ(inputs);
        return activation.activate(z);
    }
    public interface Activation{
        double activate(double z);
        double derivative(double z);
    }
    public static class Sigmoid implements Activation{
        @Override
        public double activate(double z){
            return 1.0/ (1.0 + Math.exp(-z));
        }
        public double derivative(double z){
            double sig = activate(z);
            return sig * (1 - sig);
        }
    }
}
