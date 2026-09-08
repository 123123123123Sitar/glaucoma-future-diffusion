CREATE TABLE `study_responses` (
	`session_id` text NOT NULL,
	`case_id` text NOT NULL,
	`case_order` integer NOT NULL,
	`authenticity_choice` text NOT NULL,
	`observed_side` text NOT NULL,
	`is_correct` integer,
	`progression_a` text NOT NULL,
	`progression_b` text NOT NULL,
	`features_json` text NOT NULL,
	`confidence` integer NOT NULL,
	`notes` text NOT NULL,
	`elapsed_seconds` integer NOT NULL,
	`submitted_at` text NOT NULL,
	PRIMARY KEY(`session_id`, `case_id`)
);
--> statement-breakpoint
CREATE TABLE `study_sessions` (
	`id` text PRIMARY KEY NOT NULL,
	`study_version` text NOT NULL,
	`reviewer_code` text,
	`specialty` text NOT NULL,
	`years_experience` integer,
	`started_at` text NOT NULL,
	`completed_at` text,
	`feedback` text
);
