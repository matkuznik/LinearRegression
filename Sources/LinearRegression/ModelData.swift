//
//  ModelData.swift
//  Dimensional
//
//  Created by Mateusz Kuznik on 18/04/2018.
//

import Foundation

public struct ModelData<T> {
    let featureWeights: [T]
    let bias: T
    
    public init(
        featureWeights: [T],
        bias: T) {
        
        self.featureWeights = featureWeights
        self.bias = bias
    }
}
