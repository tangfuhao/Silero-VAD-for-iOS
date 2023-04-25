//
//  ModelHandler.swift
//  Silero-VAD-for-iOS
//
//  Created by fuhao on 2023/4/23.
//

import Foundation
import Accelerate
import AVFoundation
import CoreImage
import Darwin
import Foundation
import UIKit
import onnxruntime_objc



// Result struct
struct Result {
    let processTimeMs: Double
    let score: Float
    let hn: ORTValue
    let cn: ORTValue
}




enum OrtModelError: Error {
    case error(_ message: String)
}

class ModelHandler: NSObject {
    // MARK: - Inference Properties
    let threadCount: Int32
    let threadCountLimit = 10
    
    // MARK: - Model Parameters
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 300
    let inputHeight = 300
    
    let lstm_unit_dimension = 64
    let lstm_unit_num = 2
    
    var _last_sr: Int = 0
    var _last_batch_size: Int = 0
    
    //The status of the LSTM layer
    var _h: ORTValue!;
    var _c: ORTValue!;
    
    
    
    
    private var session: ORTSession
    private var env: ORTEnv
    
    init?(modelFilename: String, modelExtension: String, threadCount: Int32 = 1) {
        guard let associateBundleURL2 = Bundle.main.url(forResource: "Silero_VAD_for_iOS", withExtension: "bundle") else {
            return nil
        }
        
        guard let podBundle = Bundle(url: associateBundleURL2) else {
            return nil
        }
        
        guard let modelPath = podBundle.path(forResource: modelFilename, ofType: modelExtension) else {
            print("Failed to get model file path with name: \(modelFilename).")
            return nil
        }
        
        
        
        self.threadCount = threadCount
        do {
            env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(ORTLoggingLevel.warning)
            try options.setIntraOpNumThreads(threadCount)
            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        } catch {
            print("Failed to create ORTSession.")
            return nil
        }
        
        super.init()
    }
    
    
    

    

    
    func parameterCheck(batchSize: Int, sr: Int) {
        guard _last_batch_size == batchSize,
              _last_sr == sr else {
            resetState(batchSize: batchSize)
            return
        }
    }
    
    func _parseToFloat(value: ORTValue?) throws -> Float {
        guard let rawOutputValue = value else {
            throw OrtModelError.error("failed to get model output")
        }
        let rawOutputData = try rawOutputValue.tensorData() as Data
        let floatValue = rawOutputData.withUnsafeBytes { $0.load(as: Float.self) }
        return floatValue
    }
    
    func _prediction(inputTensors: [ORTValue]) throws -> Result {
        let inputNames = ["input", "sr", "h", "c"]
        let outputNames: Set<String> = ["output", "hn", "cn"]
        
        guard inputTensors.count == inputNames.count else {
            throw OrtModelError.error("inputTensors.count != inputNames.count")
        }
        
        let inputDic = Dictionary(uniqueKeysWithValues: zip(inputNames, inputTensors))
        
        let interval: TimeInterval
        let startDate = Date()
        let outputs:[String: ORTValue] = try session.run(withInputs: inputDic,
                                      outputNames: outputNames,
                                      runOptions: nil)
        interval = Date().timeIntervalSince(startDate) * 1000
        
        
        let score = try _parseToFloat(value: outputs["output"])
        
        guard let hn:ORTValue = outputs["hn"],
              let cn:ORTValue = outputs["cn"] else {
            throw OrtModelError.error("hn cn is not exist")
        }
        
        // Return ORT SessionRun result
        return Result(processTimeMs: interval, score: score, hn: hn, cn: cn)
    }
    
    
    

    
}




extension ModelHandler {
    //sr: 8k or 16k
    func prediction(x: Data, sr: Int64) -> Float{
        do {
            let size = x.count / MemoryLayout<Float>.size
            let inputShape: [NSNumber] = [batchSize as NSNumber,
                                          size as NSNumber]
            let xTensor:ORTValue = try ORTValue(tensorData: NSMutableData(data: x),
                                           elementType: ORTTensorElementDataType.float,
                                           shape: inputShape)
//            let outputTensor = try xTensor.tensorData()
//            print(outputTensor.description)
            
        
            let inputShape2: [NSNumber] = []
            let srData = withUnsafeBytes(of: sr) { Data($0) }
            let srTensor:ORTValue = try ORTValue(tensorData: NSMutableData(data: srData), elementType: .int64, shape: inputShape2)
            
            let inputTensors:[ORTValue] = [xTensor, srTensor, _h, _c]
            let predictionResult = try _prediction(inputTensors: inputTensors)
            
            _h = predictionResult.hn
            _c = predictionResult.cn
            return predictionResult.score
        } catch {
            print("Unknown error: \(error)")
        }
        
        return 0
    }
    
    func resetState(batchSize: Int = 1){
        _last_sr = 0
        _last_batch_size = 0
        let inputShape: [NSNumber] = [lstm_unit_num as NSNumber,
                                      batchSize as NSNumber,
                                      lstm_unit_dimension as NSNumber]
        
        let dataCount = inputShape.reduce(1, { $0 * ($1 as! Int) })
        let zeroData = Data(repeating: 0, count: dataCount * MemoryLayout<Float>.size)
        
        _h = try! ORTValue(tensorData: NSMutableData(data: zeroData),
                                    elementType: ORTTensorElementDataType.float,
                                    shape: inputShape)
        _c = try! ORTValue(tensorData: NSMutableData(data: zeroData),
                                    elementType: ORTTensorElementDataType.float,
                                    shape: inputShape)
    }
    
}
