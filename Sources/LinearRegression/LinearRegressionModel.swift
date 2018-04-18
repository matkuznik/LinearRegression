//
//  LinearRegressionModel.swift
//  Dimensional
//
//  Created by Mateusz Kuznik on 18/04/2018.
//

import Foundation
import Dimensional

public protocol LinearRegressionModel {
    func predict(features: Matrix<Double>) -> [Double]
}

struct LearningLinearRegressionModel: LinearRegressionModel {
    let model: ModelData<Double>
    
    public func predict(features: Matrix<Double>) -> [Double] {
        let product = features * model.featureWeights
        return product.map { $0 + model.bias }
    }
}

public struct LearnedLinearRegressionModel: LinearRegressionModel {
    let model: ModelData<Double>
    let normalizator: Normalizator
    
    public func predict(features: Matrix<Double>) -> [Double] {
        let normalizedFeatures = normalizator.normalie(features)
        let product = normalizedFeatures * model.featureWeights
        return product.map { $0 + model.bias }
    }
}
