//
//  MoebiusDemoTests.swift
//  MoebiusDemoTests
//
//  Created by Kyle Howells on 2026-06-24.
//

import XCTest
import UIKit
import MoebiusMLX
@testable import MoebiusDemo

final class MoebiusDemoTests: XCTestCase {

	@MainActor
	func testImportedImageStartsWithEmptyEditableMask() throws {
		let image: UIImage = Self.makeTopBottomImage()
		let sample: DemoFixture.Sample = try XCTUnwrap(DemoFixture.makeSample(from: image))
		let expectedPixelCount: Int = MoebiusImageTensors.imageSize * MoebiusImageTensors.imageSize

		XCTAssertEqual(sample.mask.count, expectedPixelCount)
		XCTAssertTrue(sample.mask.allSatisfy { value in
			value == 0
		})
	}

	@MainActor
	func testImportedImageKeepsTopBottomOrientation() throws {
		let image: UIImage = Self.makeTopBottomImage()
		let sample: DemoFixture.Sample = try XCTUnwrap(DemoFixture.makeSample(from: image))
		let size: Int = MoebiusImageTensors.imageSize

		let topPixel: Int = ((size / 8) * size) + (size / 2)
		let bottomPixel: Int = (((size * 7) / 8) * size) + (size / 2)

		XCTAssertGreaterThan(sample.image[(topPixel * 3) + 0], 0.8)
		XCTAssertLessThan(sample.image[(topPixel * 3) + 2], -0.8)
		XCTAssertLessThan(sample.image[(bottomPixel * 3) + 0], -0.8)
		XCTAssertGreaterThan(sample.image[(bottomPixel * 3) + 2], 0.8)
	}

	private static func makeTopBottomImage() -> UIImage {
		let width: Int = 16
		let height: Int = 16
		var rgba: [UInt8] = Array(repeating: 255, count: width * height * 4)
		for y in 0 ..< height {
			for x in 0 ..< width {
				let offset: Int = ((y * width) + x) * 4
				if y < height / 2 {
					rgba[offset + 0] = 255
					rgba[offset + 1] = 0
					rgba[offset + 2] = 0
				}
				else {
					rgba[offset + 0] = 0
					rgba[offset + 1] = 0
					rgba[offset + 2] = 255
				}
			}
		}

		let data: Data = Data(rgba)
		let provider: CGDataProvider = CGDataProvider(data: data as CFData)!
		let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
		let image: CGImage = CGImage(
			width: width,
			height: height,
			bitsPerComponent: 8,
			bitsPerPixel: 32,
			bytesPerRow: width * 4,
			space: colorSpace,
			bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
			provider: provider,
			decode: nil,
			shouldInterpolate: false,
			intent: .defaultIntent
		)!
		return UIImage(cgImage: image)
	}

}
