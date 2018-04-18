//
//  TrainingData.swift
//  Dimensional
//
//  Created by Mateusz Kuznik on 31/03/2018.
//

import Foundation
import Dimensional

public struct TrainingDataInitializationError: Error {
    let message: String
}

public struct TrainingData<T: FloatingPoint> {
    let features: Matrix<T>
    let expectedResults: [T]
    
    public init(
        features: Matrix<T>,
        expectedResults: [T]) throws {
        
        if features.count == 0 {
            throw TrainingDataInitializationError(message: "features.count has to be greater than 0")
        }
        if features.count != expectedResults.count {
            throw TrainingDataInitializationError(message: "count of features and expectedResults have to be equal")
        }
        
        self.features = features
        self.expectedResults = expectedResults
        
    }
}
