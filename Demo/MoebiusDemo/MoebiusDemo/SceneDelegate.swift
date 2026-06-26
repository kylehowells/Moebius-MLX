//
//  SceneDelegate.swift
//  MoebiusDemo
//
//

import UIKit

// MARK: - SceneDelegate

final class SceneDelegate: UIResponder, UIWindowSceneDelegate {
	
	// MARK: Properties

	var window: UIWindow? = nil

	// MARK: Scene Lifecycle
	
	func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
		guard let windowScene: UIWindowScene = scene as? UIWindowScene else { return }

		let window: UIWindow = .init(windowScene: windowScene)
		window.rootViewController = ViewController()
		self.window = window
		window.makeKeyAndVisible()
	}

}
