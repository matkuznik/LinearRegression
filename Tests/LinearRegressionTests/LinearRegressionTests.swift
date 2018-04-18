import XCTest
import LinearRegression
import MathTools
import Dimensional

final class LinearRegressionTests: XCTestCase {
    
    func testThatTrainData_whenUsingNormalizedData_willProduceValidModel() throws {
        let features = Matrix( arrayLiteral:
            [22.0], [22.0], [35.0], [35.0], [36.0], [36.0], [37.0], [38.0], [38.0],
            [39.0], [41.0], [41.0], [45.0], [48.0], [49.0], [50.0], [50.0], [50.0],
            [51.0], [52.0], [54.0], [54.0], [54.0], [54.0], [56.0], [56.0], [59.0],
            [60.0], [62.0], [62.0], [67.0], [69.0], [74.0], [75.0], [75.0], [78.0])
        
        let expectedResults = [
            164.0, 380.0, 175.0, 320.0, 199.0, 198.0, 320.0, 210.0, 199.0, 295.0, 315.0,
            299.0, 315.0, 359.0, 285.0, 279.0, 409.0, 286.0, 330.0, 380.0, 375.0, 396.0,
            430.0, 299.0, 387.0, 465.0, 285.0, 295.0, 485.0, 314.0, 500.0, 390.0, 459.0,
            498.0, 460.0, 55.0
        ]

        let trainingData = try TrainingData(features: features, expectedResults: expectedResults)
        
        let linearRegression = LinearRegression()
        
        let model = linearRegression.train(
            data: trainingData,
            normalizator: StandardNormalizator(),
            verbose: false)
        
        let predictedData = model.predict(features: features)
        
        let expectedYValues = [
            233.44140536, 233.44140536, 276.34782116, 276.34782116, 279.64831469,
            279.64831469, 282.94880821, 286.24930173, 286.24930173, 289.54979526,
            296.1507823, 296.1507823, 309.35275639, 319.25423696, 322.55473049,
            325.85522401, 325.85522401, 325.85522401, 329.15571753, 332.45621106,
            339.0571981, 339.0571981, 339.0571981, 339.0571981, 345.65818515,
            345.65818515, 355.55966572, 358.86015924, 365.46114629, 365.46114629,
            381.9636139, 388.56460095, 405.06706856, 408.36756209, 408.36756209,
            418.26904265]
        
        zip(predictedData, expectedYValues).forEach { (touple) in
            let (value, expected) = touple
            XCTAssertEqual(value, expected, accuracy: 0.00000001)
        }
    }
    
    func testThatTrainData_whenUsingNormalizedDataByStandardScore_willProduceValidModel() throws {
        let features = Matrix( arrayLiteral:
            [22.0], [22.0], [35.0], [35.0], [36.0], [36.0], [37.0], [38.0], [38.0],
            [39.0], [41.0], [41.0], [45.0], [48.0], [49.0], [50.0], [50.0], [50.0],
            [51.0], [52.0], [54.0], [54.0], [54.0], [54.0], [56.0], [56.0], [59.0],
            [60.0], [62.0], [62.0], [67.0], [69.0], [74.0], [75.0], [75.0], [78.0])
        
        let expectedResults = [
            164.0, 380.0, 175.0, 320.0, 199.0, 198.0, 320.0, 210.0, 199.0, 295.0, 315.0,
            299.0, 315.0, 359.0, 285.0, 279.0, 409.0, 286.0, 330.0, 380.0, 375.0, 396.0,
            430.0, 299.0, 387.0, 465.0, 285.0, 295.0, 485.0, 314.0, 500.0, 390.0, 459.0,
            498.0, 460.0, 55.0
        ]

        let trainingData = try TrainingData(features: features, expectedResults: expectedResults)
        
        let linearRegression = LinearRegression()
        
        let model = linearRegression.train(
            data: trainingData,
            normalizator: StandardScoreNormalizator(),
            verbose: false)
        
        let predictedData = model.predict(features: features)
        
        let expectedYValues = [
            233.44140536, 233.44140536, 276.34782116, 276.34782116, 279.64831469,
            279.64831469, 282.94880821, 286.24930173, 286.24930173, 289.54979526,
            296.1507823, 296.1507823, 309.35275639, 319.25423696, 322.55473049,
            325.85522401, 325.85522401, 325.85522401, 329.15571753, 332.45621106,
            339.0571981, 339.0571981, 339.0571981, 339.0571981, 345.65818515,
            345.65818515, 355.55966572, 358.86015924, 365.46114629, 365.46114629,
            381.9636139, 388.56460095, 405.06706856, 408.36756209, 408.36756209,
            418.26904265]
        
        zip(predictedData, expectedYValues).forEach { (touple) in
            let (value, expected) = touple
            XCTAssertEqual(value, expected, accuracy: 0.0001)
        }
    }


    
    //MARK: - helpers
    private func standardScoreNormalized(_ matrix: Matrix<Double>) -> Matrix<Double> {
        let normalizedColumns: [[Double]] = matrix.columns.map { $0.standardScoreNormalized() }
        let normalizedMatrix = Matrix(normalizedColumns).transposed
        return normalizedMatrix
    }
    
    private func normalize(_ matrix: Matrix<Double>) -> Matrix<Double> {
        let normalizedColumns: [[Double]] = matrix.columns.map { $0.normalized() }
        let normalizedMatrix = Matrix(normalizedColumns).transposed
        return normalizedMatrix
    }

}

extension LinearRegressionTests {
    static var allTests = [
        ("testThatTrainData_whenUsingNormalizedData_willProduceValidModel", testThatTrainData_whenUsingNormalizedData_willProduceValidModel),
        ("testThatTrainData_whenUsingNormalizedDataByStandardScore_willProduceValidModel", testThatTrainData_whenUsingNormalizedDataByStandardScore_willProduceValidModel)
    ]
}
