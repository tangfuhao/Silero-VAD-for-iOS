//
//  VoiceActivityDetector.swift
//  Silero-VAD-for-iOS
//
//  Created by fuhao on 2023/4/24.
//

import Foundation
import AVFAudio
import onnxruntime_objc

enum DetectMode {
    case Chunk
    case Stream(windowSampleNums: Int)
}

public struct VADResult {
    public var score: Float, start: Int, end: Int
}

public struct VADTimeResult {
    public var start: Int = 0
    public var end: Int = 0
}


extension Data {
    func floatArray() -> [Float] {
        var floatArray = [Float](repeating: 0, count: self.count/MemoryLayout<Float>.stride)
        _ = floatArray.withUnsafeMutableBytes {
            self.copyBytes(to: $0, from: 0..<count)
        }
        return floatArray
    }
    
    enum Endianess {
            case little
            case big
        }
    func toFloat(endianess: Endianess = .little) -> Float? {
        guard self.count <= 4 else { return nil }
        switch endianess {
        case .big:
            let data = [UInt8](repeating: 0x00, count: 4-self.count) + self
            return data.withUnsafeBytes { $0.load(as: Float.self) }
        case .little:
            let data = self + [UInt8](repeating: 0x00, count: 4-self.count)
            return data.reversed().withUnsafeBytes { $0.load(as: Float.self) }
        }
    }
}

public class VoiceActivityDetector {
    private var _modelHandler: ModelHandler?
    private let expectedFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)
    private var _detectMode: DetectMode = .Chunk
    
    public init() {
        loadModel()
    }
    
    
    
    private func loadModel() {
        guard _modelHandler == nil else {
            return
        }
        
        _modelHandler = ModelHandler(modelFilename: "silero_vad_cache", modelExtension: "onnx", threadCount: 4)
    }
    
    private func _checkAudioFormat(pcmFormat: AVAudioFormat) -> Bool {
        // 检查采样率是否匹配
        guard pcmFormat.sampleRate == expectedFormat!.sampleRate else {
            return false
        }
        
        // 检查通道数是否匹配
        guard pcmFormat.channelCount == expectedFormat!.channelCount else {
            return false
        }
        
        // 检查位深度是否匹配
        guard pcmFormat.commonFormat == expectedFormat!.commonFormat else {
            return false
        }
        
        return true
    }
    
    func divideIntoSegments(_ x: Int, step: Int) -> [(start: Int, count: Int)] {
        var result: [(start: Int, count: Int)] = []
        var remaining = x
        var start = 0
        
        while remaining > 0 {
            let count = min(step, remaining)
            result.append((start, count))
            remaining -= count
            start += count
        }
        
        return result
    }
    
//    fileprivate func convertBigEndianToLittleEndian(_ buffer: UnsafeMutablePointer<Float32>, count: Int) -> Data {
//        var newData = Data(count: count * MemoryLayout<Float32>.size)
//
//        for i in 0..<count {
//            let value = buffer.byte
//            let offset = i * MemoryLayout<Float32>.size
//            newData.replaceSubrange(offset..<offset+MemoryLayout<Float32>.size, with: withUnsafeBytes(of: value) { Array($0) })
//
//        }
//
//        return newData
//    }
    
    fileprivate func _detectVAD(_ buffer: AVAudioPCMBuffer, _ windowSampleNums: Int, _ modelHandler: ModelHandler ) -> [VADResult]  {
        var scores: [VADResult] = []
        let channelData: UnsafePointer<UnsafeMutablePointer<Float32>> = buffer.floatChannelData!
        let channelPointer: UnsafeMutablePointer<Float32> = channelData[0]
        let frameLength = Int(buffer.frameLength)
        
        let segments = divideIntoSegments(frameLength, step: windowSampleNums)
        
        var tempCount = 0
        segments.forEach { (start: Int, count: Int) in
            let pointer: UnsafeMutablePointer<Float32> = channelPointer.advanced(by: start)
            
//            let dddd = convertBigEndianToLittleEndian(pointer, count: 512)
            
            let byteSize = count * MemoryLayout<Float32>.stride
            var data = Data(bytes: pointer, count: byteSize)
            tempCount += count
            if count < windowSampleNums {
                data.append(Data(repeating: 0, count: windowSampleNums - count))
            }

            let score = modelHandler.prediction(x: data, sr: 16000)
            scores.append(VADResult(score: score, start: start, end: tempCount-1))
        }
        
        return scores
    }
}


public extension VoiceActivityDetector {
    func resetState() {
        guard let modelHandler = _modelHandler else {
            return
        }
        _detectMode = .Chunk
        modelHandler.resetState()
    }
    
    func detect(buffer: AVAudioPCMBuffer, windowSampleNums: Int = 512) -> [VADResult]? {
        guard let modelHandler = _modelHandler else {
            return nil
        }
        guard _checkAudioFormat(pcmFormat: buffer.format) else {
            return nil
        }
        resetState()
        return _detectVAD(buffer, windowSampleNums, modelHandler)
    }
    
    func detectContinuously(buffer: AVAudioPCMBuffer, windowSampleNums: Int = 512) -> [VADResult]? {
        guard let modelHandler = _modelHandler else {
            return nil
        }
        guard _checkAudioFormat(pcmFormat: buffer.format) else {
            return nil
        }
        
        switch _detectMode {
        case .Stream(windowSampleNums: windowSampleNums):
            break
        default:
            _detectMode = .Stream(windowSampleNums: windowSampleNums)
            modelHandler.resetState()
            break
        }
        
        return _detectVAD(buffer, windowSampleNums, modelHandler)
    }
    
    
    /**
     Parameters
     ----------
     threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
        语音阈值。Silero VAD为每个音频块输出语音概率，概率高于此值则被认为是语音。
        最好为每个数据集调整此参数，但对大多数数据集来说，使用0.5是比较好的选择。
     
     minSpeechDurationInMS: int (default - 250 milliseconds)
        Final speech chunks shorter minSpeechDurationInMS are thrown out
        被分割的语音块的持续时间小于最小语音持续时间则将其丢弃。
     
     maxSpeechDurationInS: int (default -  inf)
        Maximum duration of speech chunks in seconds
        Chunks longer than maxSpeechDurationInS will be split at the timestamp of the last silence that lasts more than 100s (if any), to prevent agressive cutting.
        Otherwise, they will be split aggressively just before maxSpeechDurationInS.
        语音块的最大持续时间，超过此时间的块将在最后一次超过100秒的静默时间戳处分割，以防止过于剧烈的切割。否则，它们将在最大语音持续时间之前进行激进的分割。
     
     minSilenceDurationInMS: int (default - 100 milliseconds)
        In the end of each speech chunk wait for minSilenceDurationInMS before separating it
        在每个语音块的末尾，等待最小静默时间以进行分割。
        
     speechPadInMS: int (default - 30 milliseconds)
         Final speech chunks are padded by speechPadInMS each side
         最终的语音块在两边分别填充语音填充时间。
     */
    func detectForTimeStemp(buffer: AVAudioPCMBuffer,
                            threshold: Float = 0.5,
                            minSpeechDurationInMS: Int = 250,
                            maxSpeechDurationInS: Float = 30,
                            minSilenceDurationInMS: Int = 100,
                            windowSampleNums: Int = 512) -> [VADTimeResult]? {
        
        let sr = buffer.format.sampleRate
        guard let vadResults = detect(buffer: buffer, windowSampleNums: windowSampleNums) else {
            return nil
        }
        
        
//        vadResults.forEach { ressss in
//            print("{'end': \(ressss.end), 'start': \(ressss.start)}")
//        }
        
        
        
        let minSpeechSamples = Int(sr * Double(minSpeechDurationInMS) * 0.001)
        let maxSpeechSamples = Int(sr * Double(maxSpeechDurationInS))
        let minSilenceSample = Int(sr * Double(minSilenceDurationInMS) * 0.001)
        let minSilenceSampleAtMaxSpeech = Int(sr * Double(0.098))


        
        var triggered = false
        var speeches = [VADTimeResult]()
        var currentSpeech = VADTimeResult()

        let neg_threshold = threshold - 0.15
        var temp_end = 0
        var prev_end = 0
        var next_start = 0
        
        
        
        for (i, speech) in vadResults.enumerated() {
            let speech_prob = speech.score
            if speech_prob >= threshold && temp_end != 0 {
                temp_end = 0
                if next_start < prev_end {
                    next_start = windowSampleNums * i
                }
            }
        
            
            if speech_prob >= threshold && !triggered {
                triggered = true
                currentSpeech.start = windowSampleNums * i
                continue
            }

            if triggered && (windowSampleNums * i) - currentSpeech.start > maxSpeechSamples {
                if prev_end != 0 {
                    currentSpeech.end = prev_end
                    speeches.append(currentSpeech)
                    currentSpeech = VADTimeResult()
                    if next_start < prev_end {
                        triggered = false
                    } else {
                        currentSpeech.start = next_start
                    }
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                } else {
                    currentSpeech.end = windowSampleNums * i
                    speeches.append(currentSpeech)
                    currentSpeech = VADTimeResult()
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                    triggered = false
                    continue
                }
            }

            if speech_prob < neg_threshold && triggered {
                if temp_end == 0 {
                    temp_end = windowSampleNums * i
                }
                if (windowSampleNums * i) - temp_end > minSilenceSampleAtMaxSpeech {
                    prev_end = temp_end
                }
                if (windowSampleNums * i) - temp_end < minSilenceSample {
                    continue
                } else {
                    currentSpeech.end = temp_end
                    if (currentSpeech.end - currentSpeech.start) > minSpeechSamples {
                        speeches.append(currentSpeech)
                    }
                    currentSpeech = VADTimeResult()
                    prev_end = 0
                    next_start = 0
                    temp_end = 0
                    triggered = false
                    continue
                }
            }
        }
        
        
        let audio_length_samples = Int(buffer.frameLength)
        if currentSpeech.start > 0 && (audio_length_samples - currentSpeech.start) > minSpeechSamples {
            currentSpeech.end = audio_length_samples
            speeches.append(currentSpeech)
        }
        
        
        
        
        
        
        
        return speeches
    }
    
    
}
