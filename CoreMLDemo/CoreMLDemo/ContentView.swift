import SwiftUI
import PhotosUI
import Vision
import CoreML

struct Prediction: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Double
}

struct ContentView: View {
    @State private var item: PhotosPickerItem?
    @State private var image: UIImage?
    @State private var predictions: [Prediction] = []
    @State private var isProcessing = false

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                Text("Классификатор изображений")
                    .font(.title2).bold()
                    .padding(.top, 20)

                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 300)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .shadow(radius: 4)
                        .padding(.horizontal)
                } else {
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color(.systemGray6))
                        .frame(height: 250)
                        .overlay(
                            VStack(spacing: 8) {
                                Image(systemName: "photo.on.rectangle.angled")
                                    .font(.system(size: 40))
                                    .foregroundStyle(.secondary)
                                Text("Фото не выбрано")
                                    .foregroundStyle(.secondary)
                            }
                        )
                        .padding(.horizontal)
                }

                if isProcessing {
                    ProgressView("Распознаём…")
                        .padding()
                }

                if !predictions.isEmpty {
                    VStack(spacing: 12) {
                        ForEach(Array(predictions.enumerated()), id: \.element.id) { index, prediction in
                            PredictionRow(rank: index + 1, prediction: prediction)
                        }
                    }
                    .padding(.horizontal)
                }

                PhotosPicker(selection: $item, matching: .images) {
                    Label("Выбрать фото", systemImage: "photo.badge.plus")
                        .font(.headline)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.accentColor)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .padding(.horizontal)
                .padding(.bottom, 30)
            }
        }
        .onChange(of: item) { _, newItem in
            Task {
                guard let data = try? await newItem?.loadTransferable(type: Data.self),
                      let ui = UIImage(data: data) else { return }
                image = ui
                predictions = []
                classify(ui)
            }
        }
    }

    func classify(_ ui: UIImage) {
        guard let cg = ui.cgImage else { return }
        isProcessing = true

        do {
            let ml = try MobileNetV2(configuration: MLModelConfiguration()).model
            let vnModel = try VNCoreMLModel(for: ml)
            let request = VNCoreMLRequest(model: vnModel) { req, _ in
                guard let results = req.results as? [VNClassificationObservation] else {
                    DispatchQueue.main.async { isProcessing = false }
                    return
                }
                let top3 = results.prefix(3).map {
                    Prediction(label: $0.identifier, confidence: Double($0.confidence))
                }
                DispatchQueue.main.async {
                    predictions = top3
                    isProcessing = false
                }
            }
            request.imageCropAndScaleOption = .centerCrop
            try VNImageRequestHandler(cgImage: cg).perform([request])
        } catch {
            isProcessing = false
            predictions = [Prediction(label: "Ошибка: \(error.localizedDescription)", confidence: 0)]
        }
    }
}

struct PredictionRow: View {
    let rank: Int
    let prediction: Prediction

    var body: some View {
        HStack {
            Text("#\(rank)")
                .font(.caption).bold()
                .foregroundStyle(.secondary)
                .frame(width: 28)

            Text(prediction.label)
                .font(.body)
                .lineLimit(1)

            Spacer()

            Text("\(Int(prediction.confidence * 100))%")
                .font(.body).bold()
                .foregroundStyle(rank == 1 ? .primary : .secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
