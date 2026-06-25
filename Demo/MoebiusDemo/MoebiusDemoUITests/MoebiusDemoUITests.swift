//
//  MoebiusDemoUITests.swift
//  MoebiusDemoUITests
//
//  Created by Kyle Howells on 2026-06-24.
//

import XCTest

final class MoebiusDemoUITests: XCTestCase {

	override func setUpWithError() throws {
		self.continueAfterFailure = false
	}

	@MainActor
	func testMainScreenLabelsExplainControls() throws {
		let app: XCUIApplication = XCUIApplication()
		app.launchEnvironment["MOEBIUS_DEMO_DISABLE_AUTORUN"] = "1"
		app.launch()

		XCTAssertTrue(app.staticTexts["Model checkpoint"].waitForExistence(timeout: 5))
		XCTAssertTrue(app.staticTexts["Model precision"].exists)
		XCTAssertTrue(app.staticTexts["Sampling controls"].exists)
		XCTAssertTrue(app.staticTexts["Mask drawing controls"].exists)
		XCTAssertTrue(app.staticTexts["Image and execution controls"].exists)
		XCTAssertTrue(app.staticTexts["Denoising steps: 8"].exists)
		XCTAssertTrue(app.staticTexts["Guidance scale: 2.0"].exists)
		XCTAssertTrue(app.staticTexts["Random seed: 7"].exists)
		XCTAssertTrue(app.staticTexts["Brush size: 32 px"].exists)
		XCTAssertTrue(app.staticTexts["Paint mask"].exists)
		XCTAssertTrue(app.buttons["Load image"].exists)
		XCTAssertTrue(app.buttons["Use sample"].exists)
		XCTAssertTrue(app.buttons["Clear mask"].exists)
		XCTAssertTrue(app.buttons["Run sample"].exists)
		XCTAssertTrue(app.buttons["Compare quants"].exists)
	}
}
