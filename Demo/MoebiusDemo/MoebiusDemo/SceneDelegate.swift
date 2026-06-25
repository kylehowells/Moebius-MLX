//
//  SceneDelegate.swift
//  MoebiusDemo
//
//

import UIKit

final class SceneDelegate: UIResponder, UIWindowSceneDelegate {

	var window: UIWindow? = nil

	func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
		guard let windowScene: UIWindowScene = scene as? UIWindowScene else { return }

		let window: UIWindow = .init(windowScene: windowScene)
		window.rootViewController = ViewController()
		self.window = window
		window.makeKeyAndVisible()
	}

}
