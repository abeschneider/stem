import PackageDescription

let package = Package(
  name: "stem",
  targets: [
    Target(name: "Tensor", dependencies: []),
    Target(name: "Op", dependencies: ["Tensor"]),
    Target(name: "MNIST", dependencies: ["Tensor", "DataLoader"]),
    Target(name: "DataLoader", dependencies: ["Tensor", "Op"]),
    Target(name: "CNNExample", dependencies: ["Tensor", "Op", "DataLoader"])
  ],
  dependencies: [
    .Package(url: "https://github.com/apple/swift-protobuf.git", Version(0,9,24)),
    .Package(url: "https://github.com/1024jp/GzipSwift.git", Version(3, 1, 2))
  ]
)
