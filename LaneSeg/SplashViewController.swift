//
//  SplashViewController.swift
//  LaneSeg
//
//  Created by Htun Nay Aung on 17/5/2025.
//
import UIKit

class SplashViewController: UIViewController {

    private let logoImageView: UIImageView = {
        let imageView = UIImageView(image: UIImage(named: "Logo"))
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        return imageView
    }()

    private let appTitleLabel: UILabel = {
        let label = UILabel()
        label.text = "Vision Track"
        label.font = UIFont.boldSystemFont(ofSize: 20)
        label.textColor = .systemBlue
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let versionLabel: UILabel = {
        let label = UILabel()
        label.text = "Version 1.0.0"
        label.font = UIFont.systemFont(ofSize: 14)
        label.textColor = .gray
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()


    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white

        view.addSubview(logoImageView)
        view.addSubview(appTitleLabel)
        view.addSubview(versionLabel)

        NSLayoutConstraint.activate([
            // Center the logo horizontally and vertically
            logoImageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            logoImageView.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: -20),
            logoImageView.widthAnchor.constraint(equalToConstant: 150),
            logoImageView.heightAnchor.constraint(equalToConstant: 150),

            // Position label under the logo
            appTitleLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            appTitleLabel.topAnchor.constraint(equalTo: logoImageView.bottomAnchor, constant: 280),
            versionLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            versionLabel.topAnchor.constraint(equalTo: appTitleLabel.bottomAnchor, constant: 10)

        ])
        
        logoImageView.alpha = 0
        appTitleLabel.alpha = 0
        versionLabel.alpha = 0


        UIView.animate(withDuration: 1.0, delay: 0, options: [.curveEaseInOut], animations: {
            self.logoImageView.alpha = 1
            self.appTitleLabel.alpha = 1
            self.versionLabel.alpha = 1

        })


        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            let mainVC = ViewController()
            mainVC.modalPresentationStyle = .fullScreen
            mainVC.modalTransitionStyle = .crossDissolve

//            self.present(mainVC, animated: true) {
                // After it's properly rotated, swap root
                DispatchQueue.main.asyncAfter(deadline: .now() + 1 ) {
                    guard let window = UIApplication.shared.windows.first else { return }
                    window.rootViewController = mainVC
                
            }
        }

    }
}
