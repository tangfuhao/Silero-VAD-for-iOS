#
# Be sure to run `pod lib lint Silero-VAD-for-iOS.podspec' to ensure this is a
# valid spec before submitting.
#
# Any lines starting with a # are optional, but their use is encouraged
# To learn more about a Podspec see https://guides.cocoapods.org/syntax/podspec.html
#


Pod::Spec.new do |s|
  s.name             = 'Silero-VAD-for-iOS'
  s.version          = '0.1.1'
  s.summary          = 'A short description of Silero-VAD-for-iOS.'

# This description is used to generate tags and improve search results.
#   * Think: What does it do? Why did you write it? What is the focus?
#   * Try to keep it short, snappy and to the point.
#   * Write the description between the DESC delimiters below.
#   * Finally, don't worry about the indent, CocoaPods strips it!

  s.description      = <<-DESC
TODO: Add long description of the pod here.
                       DESC

  s.homepage         = 'https://github.com/tangfuhao/Silero-VAD-for-iOS'
  # s.screenshots     = 'www.example.com/screenshots_1', 'www.example.com/screenshots_2'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'fuhao' => 'fangshiyu2@gmail.com' }
  s.source           = { :git => 'git@github.com:tangfuhao/Silero-VAD-for-iOS.git', :tag => s.version.to_s }
  # s.social_media_url = 'https://twitter.com/<TWITTER_USERNAME>'

  s.ios.deployment_target = '13.0'

  s.source_files = 'Silero-VAD-for-iOS/Classes/**/*'
  s.static_framework = true
  s.vendored_frameworks = 'Silero-VAD-for-iOS/Frameworks/**/*.framework'
  s.resource_bundles = {
   'Silero_VAD_for_iOS' => ['Silero-VAD-for-iOS/Assets/**/*.{onnx}']
  }
  s.pod_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64' }
  s.user_target_xcconfig = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64' }

  # s.public_header_files = 'Pod/Classes/**/*.h'
  # s.frameworks = 'UIKit', 'MapKit'
  s.dependency 'onnxruntime-c'
end
