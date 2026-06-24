//
//  SceneDelegate.swift
//  MoebiusDemo
//
//  Created by Kyle Howells on 2026-06-24.
//

import UIKit

final class SceneDelegate: UIResponder, UIWindowSceneDelegate {
	var window: UIWindow?

	func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
		guard let windowScene: UIWindowScene = scene as? UIWindowScene else { return }

		let window: UIWindow = UIWindow(windowScene: windowScene)
		window.rootViewController = ViewController()
		self.window = window
		window.makeKeyAndVisible()
	}
}
