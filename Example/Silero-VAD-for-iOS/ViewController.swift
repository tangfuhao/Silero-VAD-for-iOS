//
//  ViewController.swift
//  Silero-VAD-for-iOS
//
//  Created by fuhao on 04/23/2023.
//  Copyright (c) 2023 fuhao. All rights reserved.
//

import UIKit
import Silero_VAD_for_iOS
import AVFAudio

class ViewController: UIViewController {
    let vad = VoiceActivityDetector()
    override func viewDidLoad() {
        super.viewDidLoad()
//        let floatValue: Float = -0.004486084
//        let intValue = floatValue.bitPattern
//        let hexString = String(format: "0x%08x", intValue)
//        print(hexString) // 输出: 0x40490fdb
        
        // Do any additional setup after loading the view, typically from a nib.
        view.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(testDetect)))
    }
    
    @objc
    func testDetect() {
        
        guard let buffer = loadAudioFile(url: Bundle.main.url(forResource: "en_example", withExtension: "wav")) else {
            return
        }
        
        guard let result = vad.detectForTimeStemp(buffer: buffer) else {
            return
        }
        
        result.forEach { result in
            let startS = Float(result.start) / 16000
            let endS = Float(result.end) / 16000
            print("start: \(startS) end:\(endS)")
        }
        
    }
    
    func loadAudioFile(url: URL?) -> AVAudioPCMBuffer? {
        guard let url = url,
              let file = try? AVAudioFile(forReading: url) else {
            return nil
        }

        let format = file.processingFormat
        let frameCount = UInt32(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return nil
        }

        do {
            try file.read(into: buffer)
            return buffer
        } catch {
            return nil
        }
    }

}

