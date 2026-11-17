import CoreML
import Foundation

struct Step: Codable {
  let iteration: Int
  let stage: String
  let seconds: Double
  let samples: Int?
  let maximum: Float?
  let error: String?
}

struct CodecCall: Decodable {
  let context: Int
  let frames: Int
  let values: [Int32]
}

struct CodecContract: Decodable {
  let codeGroups: Int
  let samplesPerFrame: Int
  let rotaryFrequencies: [Float]
  let rotaryScaling: Float
  let attentionWindow: Int
}

@main struct Probe {
  static func main() async throws {
    let args = CommandLine.arguments
    let report = URL(fileURLWithPath: args[3])
    let calls = try JSONDecoder().decode(
      [CodecCall].self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let modelURL = URL(fileURLWithPath: args[1])
    let contract = try decoder.decode(
      CodecContract.self,
      from: Data(contentsOf: modelURL.deletingPathExtension().appendingPathExtension("contract.json")))
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    if args.count > 6 && args[6] == "static" {
      var hints = MLOptimizationHints()
      hints.reshapeFrequency = .infrequent
      hints.specializationStrategy = .fastPrediction
      configuration.optimizationHints = hints
    }
    var steps: [Step] = []
    func save() throws { try JSONEncoder().encode(steps).write(to: report, options: .atomic) }
    var resident: [MLModel] = []
    var states: [MLState] = []
    var stateful: [(MLModel, MLState, MLDictionaryFeatureProvider)] = []
    defer { withExtendedLifetime((resident, states)) {} }
    if args.count > 4 {
      let list = URL(fileURLWithPath: args[4])
      let paths = try JSONDecoder().decode([String].self, from: Data(contentsOf: list))
      for path in paths {
        let start = Date()
        let held = try MLModel(contentsOf: URL(fileURLWithPath: path), configuration: configuration)
        resident.append(held)
        if !held.modelDescription.stateDescriptionsByName.isEmpty {
          let state = held.makeState()
          states.append(state)
          var inputs: [String: MLFeatureValue] = [:]
          for (name, description) in held.modelDescription.inputDescriptionsByName {
            let constraint = description.multiArrayConstraint!
            let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
            array.withUnsafeMutableBytes { ptr, strides in
              let data = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
              let shape = array.shape.map { $0.intValue }
              for flat in 0..<array.count {
                var remainder = flat
                var offset = 0
                for axis in shape.indices.reversed() {
                  offset += (remainder % shape[axis]) * strides[axis]
                  remainder /= shape[axis]
                }
                data[offset] =
                  name == "cosine" ? 1 : (name == "mask" && flat != array.count - 1 ? -65504 : 0)
              }
            }
            inputs[name] = MLFeatureValue(multiArray: array)
          }
          stateful.append((held, state, try MLDictionaryFeatureProvider(dictionary: inputs)))
        }
        steps.append(
          Step(
            iteration: -1, stage: "resident-" + path, seconds: Date().timeIntervalSince(start),
            samples: nil, maximum: nil, error: nil))
        try save()
      }
    }
    let load = Date()
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    var codecState = model.makeState()
    var context: Int?
    var consumed = 0

    func rotaryArray(_ values: [Float16]) throws -> MLMultiArray {
      let array = try MLMultiArray(
        shape: [1, 1, 1, NSNumber(value: values.count)], dataType: .float16)
      array.withUnsafeMutableBytes { ptr, strides in
        let data = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
        for index in values.indices { data[index * strides[3]] = values[index] }
      }
      return array
    }
    steps.append(
      Step(
        iteration: -1, stage: "load", seconds: Date().timeIntervalSince(load), samples: nil,
        maximum: nil, error: nil))
    try save()
    for (i, call) in calls.enumerated() {
      let transformerStart = Date()
      if args.count > 5 {
        let rounds = Int(args[5])!
        for _ in 0..<rounds {
          for (held, state, features) in stateful {
            let output = try await held.prediction(from: features, using: state)
            withExtendedLifetime(output) {}
          }
        }
      }
      steps.append(
        Step(iteration: i, stage: "transformers", seconds: Date().timeIntervalSince(transformerStart),
             samples: nil, maximum: nil, error: nil))
      let start = Date()
      do {
        let result: Step = try autoreleasepool {
          if call.context != context {
            context = call.context
            consumed = 0
            codecState = model.makeState()
          }
          guard call.frames >= consumed,
                call.values.count == contract.codeGroups * call.frames else {
            throw CocoaError(.fileReadCorruptFile)
          }
          let appended = call.frames - consumed
          var maximum: Float = 0
          for position in consumed..<call.frames {
            let codes = try MLMultiArray(
              shape: [1, NSNumber(value: contract.codeGroups), 1], dataType: .int32)
            codes.withUnsafeMutableBytes { ptr, strides in
              let data = ptr.baseAddress!.assumingMemoryBound(to: Int32.self)
              for group in 0..<contract.codeGroups {
                data[group * strides[1]] = call.values[group * call.frames + position]
              }
            }
            var cosine: [Float16] = []
            var sine: [Float16] = []
            for frequency in contract.rotaryFrequencies {
              let phase = frequency * Float(position)
              cosine.append(Float16(cos(phase) * contract.rotaryScaling))
              sine.append(Float16(sin(phase) * contract.rotaryScaling))
            }
            cosine += cosine
            sine += sine
            let mask = (0..<contract.attentionWindow).map { index in
              index < contract.attentionWindow - min(position + 1, contract.attentionWindow)
                ? -Float16.greatestFiniteMagnitude : Float16(0)
            }
            let features = try MLDictionaryFeatureProvider(dictionary: [
              "codes": MLFeatureValue(multiArray: codes),
              "cosine": MLFeatureValue(multiArray: rotaryArray(cosine)),
              "sine": MLFeatureValue(multiArray: rotaryArray(sine)),
              "mask": MLFeatureValue(multiArray: rotaryArray(mask)),
            ])
            let output = try model.prediction(from: features, using: codecState)
            guard let audio = output.featureValue(for: "audio")?.multiArrayValue,
                  audio.count == contract.samplesPerFrame,
                  audio.strides.last!.intValue == 1, audio.dataType == .float32 else {
              throw CocoaError(.coderReadCorrupt)
            }
            let values = audio.dataPointer.assumingMemoryBound(to: Float.self)
            for index in 0..<audio.count {
              let value = values[index]
              guard value.isFinite else { throw CocoaError(.coderReadCorrupt) }
              maximum = max(maximum, abs(value))
            }
          }
          consumed = call.frames
          return Step(
            iteration: i, stage: "predict", seconds: Date().timeIntervalSince(start),
            samples: appended * contract.samplesPerFrame, maximum: maximum, error: nil)
        }
        steps.append(result)
        try save()
      } catch {
        steps.append(
          Step(
            iteration: i, stage: "error", seconds: Date().timeIntervalSince(start), samples: nil,
            maximum: nil, error: String(reflecting: error)))
        try save()
        throw error
      }
    }
  }
}
