import SwiftUI
import PhotosUI
import Vision
import CoreML

struct ContentView: View {
    @State private var item: PhotosPickerItem?
    @State private var image: UIImage?
    @State private var result = "Выбери картинку"

    var body: some View {
        VStack(spacing: 16) {
            if let image {
                Image(uiImage: image)
                    .resizable().scaledToFit().frame(maxHeight: 300)
            }
            Text(result).font(.headline).multilineTextAlignment(.center)
            PhotosPicker("Выбрать фото", selection: $item, matching: .images)
        }
        .padding()
        .onChange(of: item) { _, newItem in
            Task {
                guard let data = try? await newItem?.loadTransferable(type: Data.self),
                      let ui = UIImage(data: data) else { return }
                image = ui
                classify(ui)
            }
        }
    }

    func classify(_ ui: UIImage) {
        guard let cg = ui.cgImage else { return }
        do {
            let ml = try MobileNetV2(configuration: MLModelConfiguration()).model
            let vnModel = try VNCoreMLModel(for: ml)
            let request = VNCoreMLRequest(model: vnModel) { req, _ in
                let top = (req.results as? [VNClassificationObservation])?.first
                DispatchQueue.main.async {
                    result = top.map { "\($0.identifier) — \(Int($0.confidence))%" }
                             ?? "Не удалось распознать"
                }
            }
            request.imageCropAndScaleOption = .centerCrop
            try VNImageRequestHandler(cgImage: cg).perform([request])
        } catch {
            result = "Ошибка: \(error.localizedDescription)"
        }
    }
}
