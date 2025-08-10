"use client";

import DashboardLayout from "@/components/dashboard-layout";
import { Card, CardContent } from "@/components/ui/card";
import { BookOpen } from "lucide-react";
import { useSearchParams } from "next/navigation";

export default function UnavailablePage() {
  const searchParams = useSearchParams();
  const label = searchParams.get("label") || "This page";

const displayLabel = label === "Dashboard" ? "Schedules" : label;

  return (
    <DashboardLayout>
      <div className="p-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-6">
          {displayLabel}
        </h1>

        <Card>
          <CardContent className="p-8 text-center">
            <BookOpen className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No {displayLabel} Available
            </h3>
            <p className="text-gray-500">
              {displayLabel} will appear here when they are added.
            </p>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
