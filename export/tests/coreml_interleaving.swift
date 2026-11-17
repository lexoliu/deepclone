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
@main struct Probe {
  static func main() async throws {
    let args = CommandLine.arguments
    let report = URL(fileURLWithPath: args[3])
    let bytes = try Data(contentsOf: URL(fileURLWithPath: args[2]))
    // The published 12 Hz codec has 16 groups; each block holds 300
    // frames plus 25 frames of left context.
    let groups = 16
    let capacity = 300 + 25
    let count = groups * capacity
    guard bytes.count % (count * 4) == 0 else { throw CocoaError(.fileReadCorruptFile) }
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
    let model = try MLModel(contentsOf: URL(fileURLWithPath: args[1]), configuration: configuration)
    steps.append(
      Step(
        iteration: -1, stage: "load", seconds: Date().timeIntervalSince(load), samples: nil,
        maximum: nil, error: nil))
    try save()
    for i in 0..<(bytes.count / (count * 4)) {
      let start = Date()
      if args.count > 5 {
        let rounds = Int(args[5])!
        for _ in 0..<rounds {
          for (held, state, features) in stateful {
            let output = try await held.prediction(from: features, using: state)
            withExtendedLifetime(output) {}
          }
        }
      }
      do {
        let result: Step = try autoreleasepool {
          let codes = try MLMultiArray(
            shape: [1, NSNumber(value: groups), NSNumber(value: capacity)], dataType: .int32)
          bytes.withUnsafeBytes { ptr in
            codes.dataPointer.copyMemory(
              from: ptr.baseAddress!.advanced(by: i * count * 4), byteCount: count * 4)
          }
          let features = try MLDictionaryFeatureProvider(dictionary: [
            "codes": MLFeatureValue(multiArray: codes)
          ])
          let output = try model.prediction(from: features)
          guard let audio = output.featureValue(for: "audio")?.multiArrayValue else {
            throw CocoaError(.coderValueNotFound)
          }
          guard audio.strides.last!.intValue == 1, audio.dataType == .float32 else {
            throw CocoaError(.coderReadCorrupt)
          }
          let values = audio.dataPointer.assumingMemoryBound(to: Float.self)
          var maximum: Float = 0
          for index in 0..<audio.count {
            let value = values[index]
            guard value.isFinite else { throw CocoaError(.coderReadCorrupt) }
            maximum = max(maximum, abs(value))
          }
          return Step(
            iteration: i, stage: "predict", seconds: Date().timeIntervalSince(start),
            samples: audio.count, maximum: maximum, error: nil)
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
