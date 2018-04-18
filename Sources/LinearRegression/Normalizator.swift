//
//  Normalizator.swift
//  LinearRegression
//
//  Created by Mateusz Kuznik on 18/04/2018.
//
//

import Foundation
import Dimensional
import MathTools

public protocol Normalizator {
    func normalie(_ data: Matrix<Double>) -> Matrix<Double>
}

public struct StandardNormalizator: Normalizator {

    public init() {}
    
    public func normalie(_ data: Matrix<Double>) -> Matrix<Double> {
        let normalizedColumns: [[Double]] = data.columns.map { $0.normalized() }
        let normalizedMatrix = Matrix(normalizedColumns).transposed
        return normalizedMatrix
    }
}

public struct StandardScoreNormalizator: Normalizator {
    
    public init() {}
    
    public func normalie(_ data: Matrix<Double>) -> Matrix<Double> {
        let normalizedColumns: [[Double]] = data.columns.map { $0.standardScoreNormalized() }
        let normalizedMatrix = Matrix(normalizedColumns).transposed
        return normalizedMatrix
    }
}
