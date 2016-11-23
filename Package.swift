import PackageDescription

let package = Package(
  name: "stem",
  targets: [
    Target(name: "Tensor", dependencies: []),
    Target(name: "Op", dependencies: ["Tensor"])
  ],
  dependencies: [
    .Package(url: "https://github.com/apple/swift-protobuf.git", Version(0,9,24))
  ]
)
