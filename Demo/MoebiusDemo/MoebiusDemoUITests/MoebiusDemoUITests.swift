//
//  MoebiusDemoUITests.swift
//  MoebiusDemoUITests
//
//

import XCTest

final class MoebiusDemoUITests: XCTestCase {

	override func setUpWithError() throws {
		self.continueAfterFailure = false
	}

	@MainActor
	func testMainScreenLabelsExplainControls() throws {
		let app: XCUIApplication = .init()
		app.launchEnvironment["MOEBIUS_DEMO_DISABLE_AUTORUN"] = "1"
		app.launch()

		XCTAssertTrue(app.staticTexts["Moebius MLX"].waitForExistence(timeout: 5))
		XCTAssertTrue(app.staticTexts["Source"].exists)
		XCTAssertTrue(app.staticTexts["Result"].exists)
		XCTAssertTrue(app.staticTexts["Run log"].exists)
		XCTAssertTrue(app.buttons["Load image"].exists)
		XCTAssertTrue(app.buttons["Generate"].exists)

		let optionsButton: XCUIElement = app.buttons["Options"]
		if optionsButton.exists, optionsButton.isHittable {
			optionsButton.tap()
		}

		XCTAssertTrue(app.staticTexts["Model"].waitForExistence(timeout: 2))
		XCTAssertTrue(app.buttons["Model checkpoint"].exists)
		XCTAssertTrue(app.staticTexts["Generation"].exists)
		XCTAssertTrue(app.staticTexts["Mask"].exists)
		XCTAssertTrue(app.staticTexts["Utilities"].exists)
		XCTAssertTrue(app.staticTexts["Denoising steps: 8"].exists)
		XCTAssertTrue(app.staticTexts["Guidance scale: 2.0"].exists)
		XCTAssertTrue(app.staticTexts["Random seed: 7"].exists)
		XCTAssertTrue(app.staticTexts["Brush size: 32 px"].exists)
		XCTAssertTrue(app.segmentedControls["Mask edit mode"].exists)
		XCTAssertTrue(app.buttons["Use sample"].exists)
		XCTAssertTrue(app.buttons["Clear mask"].exists)
		XCTAssertTrue(app.buttons["Compare quants"].exists)
	}
}
