import Foundation
import Dimensional

public class LinearRegression {
    
    public init () {}
    
    private func initialData(numberOfFeatures: Int) -> ModelData<Double> {
        return ModelData(
            featureWeights: [Double](repeating: 0.0, count: numberOfFeatures),
            bias: 0)
    }
    
    private func calculateError(forPrediction prediction: [Double], expectedResults: [Double]) -> Double {
        
        let error = zip(prediction, expectedResults).map {
            pow(($0.0 - $0.1), 2)
            }
            .reduce(Double(0), +)
        
        return error / Double(prediction.count)
    }
    
    public func train(
        data: TrainingData<Double>,
        learningRate: Double = 0.01,
        iteratons: Int = 7000,
        normalizator: Normalizator = StandardNormalizator(),
        verbose: Bool = false) -> LinearRegressionModel {
        
        let logMessage = verbose ? logDebugMessage : nil
        
        let features = normalizator.normalie(data.features)
        
        logMessage?("Features: \(features)") { true }
        var trainingModelData = initialData(numberOfFeatures: data.features.columns.count)
        
        var predictedData = LearningLinearRegressionModel(model: trainingModelData).predict(features: features)
        var error = calculateError(forPrediction: predictedData, expectedResults: data.expectedResults)
        logMessage?("Initial error: \(error)") { true }
        
        for i in 0 ..< iteratons {
            
            let derivativesOfWeight = data.features.columns
                .map { (featureWeights) -> Double in
                    let factor = (2.0 / Double(predictedData.count))
                    let sum = zip(zip(predictedData, data.expectedResults), featureWeights)
                        .reduce(Double(0)) { (result, mseParameters) -> Double in
                            // TODO: Make it more readable 🤔🙏🤷‍♂️
                            // result = sum
                            // mseParameters.0.0 - hypothesis
                            // mseParameters.0.1 - expected value
                            // mseParameters.1 - current x value
                            result + (mseParameters.0.0 - mseParameters.0.1) * mseParameters.1
                    }
                    return factor * sum //derivative of weight
            }
            
            let updatedWeights = zip(trainingModelData.featureWeights, derivativesOfWeight)
                .map { (oldFeatureWeight, derivativeOfWeight) -> Double in
                    oldFeatureWeight - derivativeOfWeight * learningRate
            }
            
            let biasSum = zip(predictedData, data.expectedResults)
                .reduce(Double(0)) { (result, mseParameters) -> Double in
                    // TODO: Make it more readable 🤔🙏🤷‍♂️
                    // result = sum
                    // mseParameters.0 - hypothesis
                    // mseParameters.1 - expected value
                    result + (mseParameters.0 - mseParameters.1)
            }
            
            let derivativesOfBias = (2.0 / Double(predictedData.count)) * biasSum
            let updatedBias = trainingModelData.bias - derivativesOfBias * learningRate
            
            trainingModelData = ModelData(featureWeights: updatedWeights, bias: updatedBias)
            predictedData = LearningLinearRegressionModel(model: trainingModelData)
                .predict(features: features)
            
            error = calculateError(forPrediction: predictedData, expectedResults: data.expectedResults)
            
            logMessage?("Epoch \(i) error: \(error)\n dw: \(derivativesOfWeight), db: \(derivativesOfBias)", { i % 250 == 0 })
        }
        
        return LearnedLinearRegressionModel(model: trainingModelData, normalizator: normalizator)
    }
    
    private func logDebugMessage(_ message: String, when condition: () -> Bool) {
        if condition() {
            print(message)
        }
    }
    
}

