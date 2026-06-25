import Darwin
import Foundation

enum DemoAutorun {
	static func write(status: String) {
		let outputStatus: String
		if let runID: String = ProcessInfo.processInfo.environment["MOEBIUS_DEMO_AUTORUN_RUN_ID"], runID.isEmpty == false {
			outputStatus = "run_id=\(runID) \(status)"
		}
		else {
			outputStatus = status
		}

		print(outputStatus)

		if let path: String = ProcessInfo.processInfo.environment["MOEBIUS_DEMO_AUTORUN_STATUS"], path.isEmpty == false {
			try? outputStatus.write(to: URL(fileURLWithPath: path), atomically: true, encoding: .utf8)
			return
		}

		guard ProcessInfo.processInfo.environment["MOEBIUS_DEMO_AUTORUN"] == "1",
		      let documentsDirectory: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
		else {
			return
		}

		let statusURL: URL = documentsDirectory.appendingPathComponent("moebius-demo-autorun-status.txt")
		try? outputStatus.write(to: statusURL, atomically: true, encoding: .utf8)
	}

	static func formatMB(_ value: Double?) -> String {
		guard let value else {
			return "unknown"
		}

		return String(format: "%.2f", value)
	}

	static func formatSeconds(_ value: TimeInterval) -> String {
		return String(format: "%.3f", value)
	}

	static func formatModelMBs(_ records: [DemoRunRecord]) -> String {
		return records.map { record in
			return "\(record.variant):\(self.formatMB(record.modelMB))"
		}
		.joined(separator: ",")
	}

	static func formatMemoryMBs(_ records: [DemoRunRecord]) -> String {
		return records.map { record in
			return "\(record.variant):\(self.formatMB(record.memoryMB))"
		}
		.joined(separator: ",")
	}

	static func formatDurations(_ records: [DemoRunRecord]) -> String {
		return records.map { record in
			return "\(record.variant):\(self.formatSeconds(record.duration))"
		}
		.joined(separator: ",")
	}

	static func residentMemoryMB() -> Double? {
		var info: mach_task_basic_info = .init()
		var count: mach_msg_type_number_t = .init(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
		let result: kern_return_t = withUnsafeMutablePointer(to: &info) { infoPointer in
			return infoPointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
				return task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), reboundPointer, &count)
			}
		}
		guard result == KERN_SUCCESS else {
			return nil
		}

		return Double(info.resident_size) / 1_000_000
	}
}
